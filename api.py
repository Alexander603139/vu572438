from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import subprocess
import httpx
from bs4 import BeautifulSoup
import asyncio
import re
from collections import defaultdict
from newspaper import Article
import feedparser
import logging
import os
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.py")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str

class SitesRequest(BaseModel):
    urls: List[str]
    max_articles_per_site: int = 5

class KeywordsRequest(BaseModel):
    text: str
    max_per_category: int = 5

@app.post("/classify")
async def classify(request: TextRequest):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "http://ai_classifier:8001/classify", json={"text": request.text}
            )
            if resp.status_code == 200:
                return resp.json()  # { "result": "...%" }
            else:
                raise HTTPException(status_code=resp.status_code, detail="AI classifier error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/keywords")
async def keywords(request: KeywordsRequest):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "http://ai_classifier:8001/extract_keywords",
                json={"text": request.text, "max_per_category": request.max_per_category}
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                raise HTTPException(status_code=resp.status_code, detail="Keyword extraction error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_text_with_newspaper(html: str, url: str) -> tuple:
    """Извлечение текста через newspaper3k + резерв BeautifulSoup"""
    try:
        article = Article(url)
        article.download(input_html=html)
        article.parse()
        text = article.text
        if text and len(text) > 200:
            preview = " ".join(text.split()[:10]) + "..."
            return text[:10000], preview
    except Exception as e:
        logger.warning(f"Newspaper failed for {url}: {e}")
    # fallback на BeautifulSoup
    try:
        soup = BeautifulSoup(html, "lxml")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        article_tag = soup.find("article")
        if article_tag:
            text = article_tag.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if len(text) > 200:
            preview = " ".join(text.split()[:10]) + "..."
            return text[:10000], preview
    except Exception as e:
        logger.warning(f"BeautifulSoup fallback failed for {url}: {e}")
    return "", "Не удалось извлечь текст"


# async def fetch_article(client, url: str):
#     try:
#         resp = await client.get(url, timeout=15.0, follow_redirects=True)
#         if resp.status_code == 200:
#             text, preview = extract_text_with_newspaper(resp.text, url)
#             if text:
#                 return {"text": text, "preview": preview, "url": url}
#     except Exception as e:
#         logger.warning(f"Fetch error {url}: {e}")
#     return None

async def fetch_article(client, url: str):
    try:
        resp = await client.get(url, timeout=15.0, follow_redirects=True)
        if resp.status_code == 200:
            text, preview = extract_text_with_newspaper(resp.text, url)
            # Если текст слишком короткий или сайт из "сложного" списка – пробуем JS
            if text and len(text) < 500:
                logger.info(f"Short text ({len(text)}) for {url}, trying JS fallback")
                js_text = await fetch_via_browser(url)
                if js_text and len(js_text) > len(text):
                    text = js_text
                    preview = ' '.join(text.split()[:10]) + '...'
            if text and len(text) > 200:
                return {'text': text[:10000], 'preview': preview, 'url': url}
    except Exception as e:
        logger.warning(f"Fetch error {url}: {e}")
    return None

async def fetch_rss_feed(feed_url: str, max_items: int):
    try:
        feed = feedparser.parse(feed_url)
        entries = feed.entries[:max_items]
        articles = []
        async with httpx.AsyncClient() as client:
            for entry in entries:
                link = entry.get("link")
                if not link:
                    continue
                try:
                    resp = await client.get(link, timeout=15.0, follow_redirects=True)
                    if resp.status_code == 200:
                        text, preview = extract_text_with_newspaper(resp.text, link)
                        if text:
                            articles.append(
                                {"text": text, "preview": preview, "url": link}
                            )
                except:
                    continue
        return articles
    except Exception as e:
        logger.warning(f"RSS error {feed_url}: {e}")
        return []


def is_likely_article(url: str, site_domain: str) -> bool:
    """Проверяет, похожа ли ссылка на статью"""
    url_lower = url.lower()
    patterns = [
        "news",
        "article",
        "post",
        "story",
        "202",
        "2025",
        "2026",
        "/p/",
        "/entry",
        "/read/",
        "/content/",
        "/story/",
        "/news/",
    ]
    if any(p in url_lower for p in patterns):
        return True
    # Проверка на наличие даты в URL (например, /2024/05/23/)
    if re.search(r"/\d{4}/\d{1,2}/\d{1,2}/", url_lower):
        return True
    return False


async def fetch_site_articles(site_url: str, max_articles: int):
    # Сначала RSS
    rss_urls = [
        site_url.rstrip("/") + "/rss",
        site_url.rstrip("/") + "/feed",
        site_url.rstrip("/") + "/rss.xml",
        site_url.rstrip("/") + "/feed.xml",
        site_url.rstrip("/") + "/news/rss",
    ]
    for rss in rss_urls:
        articles = await fetch_rss_feed(rss, max_articles)
        if articles:
            return articles[:max_articles]
    # Затем собираем ссылки с главной
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(site_url, timeout=10.0, follow_redirects=True)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "lxml")
            links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = site_url.rstrip("/") + href
                if href.startswith("http") and site_url.split("/")[2] in href:
                    if is_likely_article(href, site_url.split("/")[2]):
                        links.add(href)
            unique_links = list(links)[: max_articles * 3]
            tasks = [fetch_article(client, link) for link in unique_links]
            results = await asyncio.gather(*tasks)
            articles = [r for r in results if r is not None]
            return articles[:max_articles]
        except Exception as e:
            logger.warning(f"Site fetch error {site_url}: {e}")
            return []

async def fetch_via_browser(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "http://browser:8002/fetch",
                json={"url": url, "timeout": 10000}
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("text", "")
            else:
                logger.warning(f"Browser service error: {resp.status_code}")
                return ""
    except Exception as e:
        logger.warning(f"Browser service exception: {e}")
        return ""

@app.post("/analyze_sites")
async def analyze_sites(request: SitesRequest):
    results = []
    for site in request.urls:
        articles_data = await fetch_site_articles(site, request.max_articles_per_site)
        if not articles_data:
            results.append(
                {
                    "url": site,
                    "error": "Не найдено статей (проверьте RSS или HTML-ссылку)",
                    "articles": [],
                }
            )
            continue
        articles_result = []
        total_counts = defaultdict(int)
        for art in articles_data:
            text = art["text"]
            preview = art["preview"]

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        "http://ai_classifier:8001/classify", json={"text": text}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        output = data.get("result", "")
                    else:
                        output = ""
            except Exception as e:
                logger.error(f"YandexGPT error for {art['url']}: {e}")
                output = ""
            dist = {}

            confidence = 0.0
            for line in output.split("\n"):
                if ":" in line and "%" in line:
                    parts = line.split(":")
                    if len(parts) == 2:
                        cat = parts[0].strip()
                        try:
                            percent = float(parts[1].replace("%", "").strip())
                            dist[cat] = percent
                            total_counts[cat] += percent
                        except:
                            pass
                elif "Уверенность ИИ:" in line:
                    # Извлекаем число после двоеточия
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except:
                        confidence = 0.0

            for line in output.split("\n"):
                if ":" in line and "%" in line:
                    parts = line.split(":")
                    if len(parts) == 2:
                        cat = parts[0].strip()
                        try:
                            percent = float(parts[1].replace("%", "").strip())
                            dist[cat] = percent
                            total_counts[cat] += percent
                        except:
                            pass
            articles_result.append(
                {
                    "url": art["url"],
                    "preview": preview,
                    "distribution": dist,
                    "confidence": confidence,
                }
            )
        if total_counts:
            total = sum(total_counts.values())
            norm = (
                {cat: round(val / total * 100, 1) for cat, val in total_counts.items()}
                if total > 0
                else {}
            )
        else:
            norm = {}

        total_confidence = sum(art.get("confidence", 0) for art in articles_result)
        avg_confidence = (
            round(total_confidence / len(articles_result), 2)
            if articles_result
            else 0.0
        )

        results.append(
            {
                "url": site,
                "articles_parsed": len(articles_data),
                "distribution": norm,
                "articles": articles_result,
                "avg_confidence": avg_confidence,
            }
        )
    return {"results": results}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
