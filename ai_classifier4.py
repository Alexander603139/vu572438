import os
import json
import logging
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class TextRequest(BaseModel):
    text: str


FOLDER_ID = os.environ.get("YC_FOLDER_ID")
API_KEY = os.environ.get("YC_API_KEY")

CLASSIFY_URL = (
    "https://llm.api.cloud.yandex.net/foundationModels/v1/fewShotTextClassification"
)
HEADERS = {"Authorization": f"Api-Key {API_KEY}", "Content-Type": "application/json"}


async def query_yandexgpt(text: str) -> dict:
    payload = {
        "modelUri": f"cls://{FOLDER_ID}/yandexgpt/rc",
        "taskDescription": "Классификация текста по политической ориентации",
        "labels": [
            "Экономические левые",
            "Экономические правые",
            "Социально-либеральные",
            "Социально-авторитарные",
        ],
        "text": text,
        "samples": [
            {
                "text": "Государство должно национализировать заводы и повысить налоги для богатых.",
                "label": "Экономические левые",
            },
            {
                "text": "Необходимо снизить налоги и приватизировать госпредприятия.",
                "label": "Экономические правые",
            },
            {
                "text": "Свобода слова и права ЛГБТ должны быть защищены законом.",
                "label": "Социально-либеральные",
            },
            {
                "text": "Традиционные ценности и суверенитет важнее западных свобод.",
                "label": "Социально-авторитарные",
            },
        ],
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(CLASSIFY_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        return resp.json()


@app.post("/classify")
async def classify(request: TextRequest):
    try:
        result = await query_yandexgpt(request.text)
        predictions = result.get("predictions", [])
        if not predictions:
            raise ValueError("Нет предсказаний")
        # predictions: список объектов {"label": "категория", "confidence": 0.99}
        # Выбираем лучший
        best = max(predictions, key=lambda x: x.get("confidence", 0))
        category = best["label"]
        confidence = best["confidence"]
        # Формируем распределение (100% для лучшей категории)
        categories = [
            "Экономические левые",
            "Экономические правые",
            "Социально-либеральные",
            "Социально-авторитарные",
        ]
        distribution = {cat: (100.0 if cat == category else 0.0) for cat in categories}
        result_str = "\nРезультат классификации:\n"
        for cat, perc in distribution.items():
            result_str += f"  {cat}: {perc}%\n"
        result_str += f"Уверенность ИИ: {confidence:.2f}\n"
        return {"result": result_str}
    except Exception as e:
        logger.exception("Error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
