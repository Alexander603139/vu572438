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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextRequest(BaseModel):
    text: str

# Переменные окружения
FOLDER_ID = os.environ.get("YC_FOLDER_ID")
API_KEY = os.environ.get("YC_API_KEY")

CLASSIFY_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/fewShotTextClassification"
HEADERS = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json"
}

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

@app.get("/health")
async def health():
    return {"status": "ok"}