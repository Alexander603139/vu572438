# browser_service.py
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playwright.async_api import async_playwright
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class FetchRequest(BaseModel):
    url: str
    timeout: int = 10000  # миллисекунды

# Семафор для ограничения числа одновременных браузеров
_semaphore = asyncio.Semaphore(2)

@app.post("/fetch")
async def fetch_text(request: FetchRequest):
    async with _semaphore:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--disable-dev-shm-usage', '--no-sandbox']
                )
                page = await browser.new_page()
                await page.goto(request.url, timeout=request.timeout, wait_until='networkidle')
                text = await page.text_content('body')
                await browser.close()
                if text:
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    text = ' '.join(lines)
                    return {"text": text[:10000]}
                else:
                    return {"text": ""}
        except Exception as e:
            logger.error(f"Playwright error for {request.url}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}