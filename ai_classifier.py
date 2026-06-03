import os
import json
import logging
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml
from pathlib import Path
import aioredis
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from fastapi.responses import Response
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
requests_total = Counter('yandexgpt_requests_total', 'Total requests to YandexGPT', ['endpoint'])
tokens_used_total = Counter('yandexgpt_tokens_used_total', 'Tokens consumed', ['endpoint'])
confidence_histogram = Histogram('yandexgpt_confidence', 'Confidence of classification', buckets=[0.5,0.7,0.8,0.9,0.95,0.99])
cache_hits = Counter('redis_cache_hits_total', 'Cache hits')
cache_misses = Counter('redis_cache_misses_total', 'Cache misses')

class TextRequest(BaseModel):
    text: str

class KeywordsRequest(BaseModel):
    text: str
    max_per_category: int = 5

# Переменные окружения
FOLDER_ID = os.environ.get("YC_FOLDER_ID")
API_KEY = os.environ.get("YC_API_KEY")

CLASSIFY_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/fewShotTextClassification"
HEADERS = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json"
}

# Загружаем маркеры из YAML (путь внутри контейнера)
MARKERS_PATH = "/app/political_markers.yml"
with open(MARKERS_PATH, 'r', encoding='utf-8') as f:
    markers_config = yaml.safe_load(f)

CATEGORY_MARKERS = {}
for cat, data in markers_config['categories'].items():
    CATEGORY_MARKERS[data['name']] = data['markers']  # русские названия категорий

# Загрузка few-shot примеров из YAML-файлов
SAMPLES_DIR = Path("/app/samples")  # внутри контейнера
samples = []
label_map = {
    "economic_left.yml": "Экономические левые",
    "economic_right.yml": "Экономические правые",
    "social_liberal.yml": "Социально-либертарные",
    "social_authoritarian.yml": "Социально-авторитарные"
}
for filename, label in label_map.items():
    filepath = SAMPLES_DIR / filename
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = yaml.safe_load(f)
            for text in texts:
                samples.append({"text": text, "label": label})
    else:
        logger.warning(f"Samples file not found: {filepath}")

# Redis connection
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_client = None

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = await aioredis.from_url(redis_url, decode_responses=True)
    logger.info("Redis connected")

@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()

async def query_yandexgpt(text: str) -> dict:
    # Проверка кэша
    cache_key = f"yandexgpt:{text}"
    cached = await redis_client.get(cache_key)
    if cached:
        logger.info("Cache hit")
        return json.loads(cached)
    logger.info("Cache miss, calling YandexGPT")
    payload = {
        "modelUri": f"cls://{FOLDER_ID}/yandexgpt/rc",
        "taskDescription": "Определи политическую ориентацию текста. Категории: Экономические левые, Экономические правые, Социально-либертарные, Социально-авторитарные.",
        "labels": [
            "Экономические левые",
            "Экономические правые",
            "Социально-либертарные",
            "Социально-авторитарные"
        ],
        "text": text,
        "samples": samples
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(CLASSIFY_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # tokens
        tokens = data.get("inputTokens", 0)
        tokens_used_total.labels(endpoint='/classify').inc(tokens)
        # confidence — берём максимальную из predictions
        predictions = data.get("predictions", [])
        if predictions:
            max_conf = max(p.get("confidence", 0) for p in predictions)
            confidence_histogram.observe(max_conf)
        # Сохраняем в кэш на 1 час (3600 секунд)
        await redis_client.setex(cache_key, 3600, json.dumps(data))
        return data

@app.post("/classify")
async def classify(request: TextRequest):
    try:
        result = await query_yandexgpt(request.text)
        predictions = result.get("predictions", [])
        if not predictions:
            raise ValueError("No predictions")
        # Формируем строку с процентами для каждой категории
        total = sum(p.get("confidence", 0) for p in predictions)
        if total == 0:
            total = 1.0
        lines = ["Результат классификации:"]
        for p in predictions:
            label = p.get("label")
            conf = p.get("confidence", 0)
            percent = round(conf / total * 100, 1)
            lines.append(f"  {label}: {percent}%")
        # Добавляем уверенность (максимальная confidence)
        max_conf = max(p.get("confidence", 0) for p in predictions)
        lines.append(f"Уверенность ИИ: {max_conf:.2f}")
        result_str = "\n".join(lines)
        return {"result": result_str}
    except Exception as e:
        logger.exception("Error")
        raise HTTPException(status_code=500, detail=str(e))

async def extract_keywords_with_categories(text: str, max_per_category: int) -> dict:
    """Отправляет запрос в YandexGPT (chat) для выделения ключевых фраз по категориям."""
    # Формируем промпт с инструкцией и списком категорий + маркеров
    categories_prompt = "\n".join([
        f"- {cat}: примеры маркеров: {', '.join(markers[:10])}..."
        for cat, markers in CATEGORY_MARKERS.items()
    ])
    prompt = f"""Ты — помощник, который выделяет ключевые политические фразы из текста. У нас есть категории с типичными маркерами:
    {categories_prompt}
    Задача: прочитай текст и для каждой категории выдели до {max_per_category} ключевых слов или коротких фраз (на русском), 
    которые встречаются в тексте и относятся к этой категории. Если для категории ничего не подходит, верни пустой список.
    Ответ должен быть строго в формате JSON, где ключи — названия категорий, значения — списки строк. 
    Пример: {{"Экономические левые": ["национализация", "социальная справедливость"], "Экономические правые": [], ...}}
    Текст для анализа:
    {text}
    """
    requests_total.labels(endpoint='/extract_keywords').inc()
    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/rc",
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 800
        },
        "messages": [
            {"role": "system", "text": "Ты — аналитик, выделяющий ключевые фразы по заданным категориям. Отвечай только JSON."},
            {"role": "user", "text": prompt}
        ]
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=HEADERS,
            json=payload
        )
        resp.raise_for_status()
        result = resp.json()
        response_text = result['result']['alternatives'][0]['message']['text']
        # Извлекаем JSON из ответа (модель может добавить пояснения)
        import json
        # Ищем первую '{' и последнюю '}'
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response_text[start:end]
            keywords = json.loads(json_str)
        else:
            keywords = {}
        # Убедимся, что все категории присутствуют
        for cat in CATEGORY_MARKERS:
            if cat not in keywords:
                keywords[cat] = []
        return keywords

@app.post("/extract_keywords")
async def extract_keywords(request: KeywordsRequest):
    try:
        keywords = await extract_keywords_with_categories(request.text, request.max_per_category)
        return {"keywords": keywords}
    except Exception as e:
        logger.exception("Error extracting keywords")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "ok"}