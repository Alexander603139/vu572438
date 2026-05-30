from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
import json
import re

app = FastAPI()

class TextRequest(BaseModel):
    text: str

# Few-shot промпт с вашими 4 категориями и примерами
SYSTEM_PROMPT = """Ты — классификатор политических текстов. Относи текст к одной из четырёх категорий:
- Экономические левые: поддержка госрегулирования, соцвыплат, национализации.
- Экономические правые: поддержка рынка, приватизации, снижения налогов.
- Социально-либертарные: поддержка свобод, прав человека, толерантности.
- Социально-авторитарные: поддержка порядка, традиций, сильной власти.

Примеры:
Текст: "Государство должно национализировать заводы и повысить налоги для богатых." → Экономические левые
Текст: "Необходимо снизить налоги и приватизировать госпредприятия." → Экономические правые
Текст: "Свобода слова и права ЛГБТ должны быть защищены законом." → Социально-либертарные
Текст: "Традиционные ценности и суверенитет важнее западных свобод." → Социально-авторитарные

Теперь классифицируй следующий текст. Ответ должен быть строго в формате JSON:
{"category": "название категории", "confidence": 0.0-1.0}"""

async def query(prompt: str) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "http://172.17.0.1:11434/api/generate",
            json={
                "model": "tinyllama:1.1b",
                "prompt": prompt,
                "stream": False,
                "system": SYSTEM_PROMPT
            }
        )
        return resp.json()

@app.post("/classify")
async def classify(request: TextRequest):
    try:
        user_prompt = f"Текст: {request.text}\nКатегория (только JSON):"
        result = await query(user_prompt)
        response_text = result.get("response", "")
        # Извлекаем JSON из ответа модели
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data
        else:
            # fallback: ищем категорию в тексте
            for cat in ["Экономические левые", "Экономические правые", "Социально-либертарные", "Социально-авторитарные"]:
                if cat in response_text:
                    return {"category": cat, "confidence": 0.5}
            raise ValueError("Не удалось распознать категорию")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)