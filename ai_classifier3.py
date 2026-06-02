# /opt/ai-agent/ai_classifier.py
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


# --- Конфигурация YandexGPT ---
# ID каталога и API-ключ сервисного аккаунта (будут читаться из переменных окружения)
FOLDER_ID = os.environ.get("YC_FOLDER_ID")
API_KEY = os.environ.get("YC_API_KEY")

# URL API классификации
CLASSIFY_URL = (
    "https://llm.api.cloud.yandex.net/foundationModels/v1/fewShotTextClassification"
)

# Заголовки авторизации
HEADERS = {"Authorization": f"Api-Key {API_KEY}", "Content-Type": "application/json"}

# Промпт, где задаются категории и примеры (few-shot)
PROMPT = """{input_text}

Определи политическую ориентацию данного текста. Относи текст к одной из следующих категорий:
1. Экономические левые (госрегулирование, соцвыплаты, национализация)
2. Экономические правые (рынок, приватизация, снижение налогов)
3. Социально-либеральные (свободы, права человека, толерантность)
4. Социально-авторитарные (порядок, традиции, сильная власть)

Примеры:
Текст: "Государство должно национализировать заводы и повысить налоги для богатых."
Категория: Экономические левые

Текст: "Необходимо снизить налоги и приватизировать госпредприятия."
Категория: Экономические правые

Текст: "Свобода слова и права ЛГБТ должны быть защищены законом."
Категория: Социально-либеральные

Текст: "Традиционные ценности и суверенитет важнее западных свобод."
Категория: Социально-авторитарные

Теперь проанализируй следующий текст и выведи только название категории из списка выше.

Текст: {input_text}
Категория:"""


async def query_yandexgpt(text: str) -> dict:
    """Отправляет запрос к YandexGPT и возвращает ответ."""
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
        response = await client.post(CLASSIFY_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()


@app.post("/classify")
async def classify(request: TextRequest):
    """Эндпоинт для классификации текста."""
    try:
        result = await query_yandexgpt(request.text)
        # Извлекаем предсказанную категорию из ответа
        # Ожидаемая структура ответа: {'predictions': [{'label': 'Категория', 'confidence': 0.99}]}
        predictions = result.get("predictions", [])
        if not predictions:
            raise ValueError("Не удалось получить ответ от YandexGPT")

        best_prediction = predictions[0]
        category = best_prediction.get("label")
        confidence = best_prediction.get("confidence", 0.0)

        # Формируем распределение в процентах
        categories = [
            "Экономические левые",
            "Экономические правые",
            "Социально-либеральные",
            "Социально-авторитарные",
        ]
        distribution = {cat: (100.0 if cat == category else 0.0) for cat in categories}

        # Формируем результат в том же формате, что и старый классификатор
        result_str = "\nРезультат классификации:\n"
        for cat, perc in distribution.items():
            result_str += f"  {cat}: {perc}%\n"
        result_str += f"Уверенность ИИ: {confidence:.2f}\n"
        return {"result": result_str}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.text}")
        raise HTTPException(
            status_code=500, detail=f"Ошибка YandexGPT API: {e.response.text}"
        )
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}
