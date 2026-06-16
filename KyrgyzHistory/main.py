import os
import json
import re
import asyncio
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI, RateLimitError, APIError
from rank_bm25 import BM25Okapi

load_dotenv()

app = FastAPI(title="Kyrgyz History AI Backend (Hybrid)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "poolside/laguna-m.1:free",
    "nex-agi/nex-n2-pro:free",
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]

answer_cache = {}
CACHE_SIZE_LIMIT = 100
sources_data = []
bm25_index = None

TOKENIZER = re.compile(r'\w+')

def tokenize(text):
    return TOKENIZER.findall(text.lower())

def load_data():
    global sources_data, bm25_index
    with open("sources.json", "r", encoding="utf-8") as f:
        sources_data = json.load(f)
    
    tokenized_corpus = [tokenize(item["text"]) for item in sources_data]
    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"✅ Загружено {len(sources_data)} источников. Индекс BM25 построен.")

load_data()

def get_cache_key(message: str, context: str) -> str:
    return hashlib.md5(f"{message}|{context}".encode()).hexdigest()

class ChatRequest(BaseModel):
    message: str

async def query_single_model(model: str, messages: list, timeout: int = 15):
    try:
        task = asyncio.create_task(
            ai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,  # Чуть выше для более гибких ответов
                max_tokens=600
            )
        )
        response = await asyncio.wait_for(task, timeout=timeout)
        print(f"✅ {model} ответил")
        return response.choices[0].message.content, model
        
    except asyncio.TimeoutError:
        print(f"⏱️ Таймаут (15с): {model}")
        return None, None
    except (RateLimitError, APIError) as e:
        print(f"⚠️ Лимит/Ошибка: {model}")
        return None, None
    except Exception as e:
        print(f"❌ Ошибка {model}: {str(e)[:50]}")
        return None, None

async def query_models_parallel(messages: list, top_n: int = 3):
    models_to_try = FREE_MODELS[:top_n]
    tasks = [asyncio.create_task(query_single_model(model, messages)) for model in models_to_try]
    
    try:
        for completed_task in asyncio.as_completed(tasks):
            answer, used_model = await completed_task
            if answer is not None:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                return answer, used_model
    except Exception as e:
        print(f"❌ Ошибка в параллельных запросах: {e}")
    
    return None, None

@app.post("/api/chat")
async def chat(req: ChatRequest):
    start_time = datetime.now()
    
    # 1. BM25 ПОИСК
    tokenized_query = tokenize(req.message)
    scores = bm25_index.get_scores(tokenized_query)
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]  # Топ-5 вместо топ-3
    
    max_score = max(scores) if len(scores) > 0 else 0
    
    # Определяем режим работы
    if max_score > 5:
        mode = "high_relevance"  # Высокая релевантность
    elif max_score > 0:
        mode = "low_relevance"   # Низкая релевантность
    else:
        mode = "no_match"        # Нет совпадений
    
    print(f"🔍 BM25: max_score={max_score:.2f}, режим={mode}")
    
    context = ""
    
    if mode == "high_relevance":
        # РЕЖИМ 1: Найдены высокорелевантные документы
        for idx in top_n_indices:
            if scores[idx] > 0:
                item = sources_data[idx]
                text_snippet = item['text'][:800] + ("..." if len(item['text']) > 800 else "")
                context += f"[{item['source']}] ({item['period']}): {text_snippet}\n"
    elif mode == "low_relevance":
        # РЕЖИМ 2: Найдены документы с низкой релевантностью
        for idx in top_n_indices:
            if scores[idx] > 0:
                item = sources_data[idx]
                context += f"[{item['source']}] ({item['period']}): {item['text'][:400]}...\n"
        
        # Добавляем обзор остальных источников
        context += "\n--- ОБЗОР ДРУГИХ ИСТОЧНИКОВ ---\n"
        for i, item in enumerate(sources_data[:15]):
            if i not in top_n_indices:
                context += f"- [{item['source']}] ({item['period']}): {item['text'][:100]}...\n"
    else:
        # РЕЖИМ 3: Нет совпадений — отправляем полный обзор
        context = "ОБЗОР ВСЕХ ИСТОЧНИКОВ В БАЗЕ ДАННЫХ:\n"
        for i, item in enumerate(sources_data[:20]):
            context += f"- [{item['source']}] ({item['period']}): {item['text'][:150]}...\n"
        
        if len(sources_data) > 20:
            context += f"\n... и еще {len(sources_data) - 20} источников.\n"

    # 2. КЭШ
    cache_key = get_cache_key(req.message, context)
    if cache_key in answer_cache:
        elapsed = (datetime.now() - start_time).total_seconds()
        cached_answer, cached_model = answer_cache[cache_key]
        return {"answer": cached_answer, "used_model": f"{cached_model} (кэш)", "response_time": f"{elapsed:.2f}с"}

    # 3. ГИБРИДНЫЙ ПРОМПТ (отвечает и по источникам, и из общих знаний)
    system_prompt = """Ты — эксперт по истории Кыргызстана. Твоя задача — отвечать на ВСЕ вопросы пользователя.

ПРИОРИТЕТ ОТВЕТА:
1. Если в предоставленных ИСТОЧНИКАХ есть ответ — используй его и укажи источник в [].
2. Если в источниках НЕТ ответа, но ты ЗНАЕШЬ ответ из общих исторических знаний — дай ответ и укажи "[Общая историческая справка]".
3. Если ты НЕ ЗНАЕШЬ ответа — честно скажи: "К сожалению, информация по этому вопросу отсутствует."

ФОРМАТ ОТВЕТА:
- Отвечай КРАТКО и по существу (2-4 предложения).
- Всегда указывай источник или пометку "[Общая историческая справка]".
- Если вопрос общий (например, "расскажи об истории"), дай структурированный обзор основных периодов.

ВАЖНО: Ты ДОЛЖЕН ответить на ЛЮБОЙ вопрос по истории Кыргызстана, даже если его нет в источниках."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"КОНТЕКСТ ИСТОЧНИКОВ:\n{context}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{req.message}"}
    ]

    # 4. ПАРАЛЛЕЛЬНЫЙ ЗАПРОС
    print(f"🚀 Запрос к {min(3, len(FREE_MODELS))} моделям...")
    answer, used_model = await query_models_parallel(messages, top_n=3)
    
    if answer is None:
        for model in FREE_MODELS[3:]:
            answer, used_model = await query_single_model(model, messages, timeout=20)
            if answer: break
    
    if answer is None:
        raise HTTPException(status_code=503, detail="Модели перегружены. Попробуйте через минуту.")
    
    # 5. КЭШИРОВАНИЕ
    if len(answer_cache) >= CACHE_SIZE_LIMIT:
        del answer_cache[next(iter(answer_cache))]
    answer_cache[cache_key] = (answer, used_model)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✅ Готово за {elapsed:.2f}с ({used_model})")
    
    return {"answer": answer, "used_model": used_model, "response_time": f"{elapsed:.2f}с"}

@app.get("/")
def health_check():
    return {"status": "Running", "sources": len(sources_data), "cache": len(answer_cache)}