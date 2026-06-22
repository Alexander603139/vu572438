import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playwright.async_api import async_playwright, Browser, Playwright
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FetchRequest(BaseModel):
    url: str
    timeout: int = 10000  # миллисекунды

# Глобальные объекты
_playwright: Playwright | None = None
_browser: Browser | None = None
_semaphore = asyncio.Semaphore(5)  # ограничим число одновременных запросов

# Селекторы для удаления мусорных блоков
NAV_SELECTORS = [
    "header", "footer", "nav", "aside",
    ".menu", ".navigation", ".sidebar", ".ad", ".banner",
    "[role='banner']", "[role='navigation']", "[role='complementary']",
    ".header", ".footer", ".nav", ".side"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _playwright, _browser
    # Старт: запускаем браузер один раз
    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(
        headless=True,
        args=['--disable-dev-shm-usage', '--no-sandbox']
    )
    logger.info("Browser launched")
    yield
    # Остановка: закрываем браузер и playwright
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()
    logger.info("Browser closed")

app = FastAPI(lifespan=lifespan)

@app.post("/fetch")
async def fetch_text(request: FetchRequest):
    async with _semaphore:
        if not _browser:
            raise HTTPException(status_code=503, detail="Browser not ready")
        
        try:
            # Создаём новую страницу в существующем браузере
            page = await _browser.new_page()
            try:
                await page.goto(request.url, timeout=request.timeout, wait_until='networkidle')
                
                # Удаляем навигационные и рекламные блоки (по селекторам)
                for selector in NAV_SELECTORS:
                    elements = await page.query_selector_all(selector)
                    for el in elements:
                        await el.evaluate("node => node.remove()")
                
                # Дополнительная очистка через JS (теги и роли)
                await page.evaluate("""() => {
                    ['header', 'footer', 'nav', 'aside', 'menu'].forEach(tag => {
                        document.querySelectorAll(tag).forEach(el => el.remove());
                    });
                    document.querySelectorAll('[role="banner"], [role="navigation"], [role="complementary"]')
                        .forEach(el => el.remove());
                }""")
                
                # Извлекаем текст из body (или main, если body пустой)
                text = await page.text_content('body')
                if not text or len(text.strip()) < 50:
                    text = await page.text_content('main') or ""
                
                # Очищаем от лишних пробелов и ограничиваем размер
                if text:
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    text = ' '.join(lines)
                    text = text[:10000]   # максимум 10k символов
                else:
                    text = ""
                
                return {"text": text}
            
            finally:
                await page.close()   # страница закрывается после каждого запроса
        
        except Exception as e:
            logger.error(f"Playwright error for {request.url}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}