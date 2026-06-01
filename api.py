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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(BASE_DIR, 'classifier.py')

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

# @app.post("/classify")
# async def classify(request: TextRequest):
#     try:
#         async with httpx.AsyncClient(timeout=120.0) as client:
#             resp = await client.post(
#                 "http://ai_classifier:8001/classify",
#                 json={"text": request.text}
#             )
#             if resp.status_code == 200:
#                 data = resp.json()
#                 category = data.get("category")
#                 confidence = data.get("confidence", 0.0)
#                 # Преобразуем единственную категорию в распределение 100%
#                 categories = ["Экономические левые", "Экономические правые", "Социально-либертарные", "Социально-авторитарные"]
#                 dist = {cat: (100.0 if cat == category else 0.0) for cat in categories}
#                 # Формируем вывод в том же формате, что и старый классификатор
#                 result_str = "\nРезультат классификации:\n"
#                 for cat, perc in dist.items():
#                     result_str += f"  {cat}: {perc}%\n"
#                 result_str += f"Уверенность ИИ: {confidence}\n"
#                 return {"result": result_str}
#             else:
#                 raise HTTPException(status_code=resp.status_code, detail="AI classifier error")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
async def classify(request: TextRequest):
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "http://ai_classifier:8001/classify",
                json={"text": request.text}
            )
            if resp.status_code == 200:
                return resp.json()  # { "result": "...%" }
            else:
                raise HTTPException(status_code=resp.status_code, detail="AI classifier error")
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
            preview = ' '.join(text.split()[:10]) + '...'
            return text[:10000], preview
    except Exception as e:
        logger.warning(f"Newspaper failed for {url}: {e}")
    # fallback на BeautifulSoup
    try:
        soup = BeautifulSoup(html, 'lxml')
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        article_tag = soup.find('article')
        if article_tag:
            text = article_tag.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        if len(text) > 200:
            preview = ' '.join(text.split()[:10]) + '...'
            return text[:10000], preview
    except Exception as e:
        logger.warning(f"BeautifulSoup fallback failed for {url}: {e}")
    return "", "Не удалось извлечь текст"

async def fetch_article(client, url: str):
    try:
        resp = await client.get(url, timeout=15.0, follow_redirects=True)
        if resp.status_code == 200:
            text, preview = extract_text_with_newspaper(resp.text, url)
            if text:
                return {'text': text, 'preview': preview, 'url': url}
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
                link = entry.get('link')
                if not link:
                    continue
                try:
                    resp = await client.get(link, timeout=15.0, follow_redirects=True)
                    if resp.status_code == 200:
                        text, preview = extract_text_with_newspaper(resp.text, link)
                        if text:
                            articles.append({'text': text, 'preview': preview, 'url': link})
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
        'news', 'article', 'post', 'story', '202', '2025', '2026',
        '/p/', '/entry', '/read/', '/content/', '/story/', '/news/'
    ]
    if any(p in url_lower for p in patterns):
        return True
    # Проверка на наличие даты в URL (например, /2024/05/23/)
    if re.search(r'/\d{4}/\d{1,2}/\d{1,2}/', url_lower):
        return True
    return False

async def fetch_site_articles(site_url: str, max_articles: int):
    # Сначала RSS
    rss_urls = [
        site_url.rstrip('/') + '/rss',
        site_url.rstrip('/') + '/feed',
        site_url.rstrip('/') + '/rss.xml',
        site_url.rstrip('/') + '/feed.xml',
        site_url.rstrip('/') + '/news/rss'
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
            soup = BeautifulSoup(resp.text, 'lxml')
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('/'):
                    href = site_url.rstrip('/') + href
                if href.startswith('http') and site_url.split('/')[2] in href:
                    if is_likely_article(href, site_url.split('/')[2]):
                        links.add(href)
            unique_links = list(links)[:max_articles * 3]
            tasks = [fetch_article(client, link) for link in unique_links]
            results = await asyncio.gather(*tasks)
            articles = [r for r in results if r is not None]
            return articles[:max_articles]
        except Exception as e:
            logger.warning(f"Site fetch error {site_url}: {e}")
            return []

@app.post("/analyze_sites")
async def analyze_sites(request: SitesRequest):
    results = []
    for site in request.urls:
        articles_data = await fetch_site_articles(site, request.max_articles_per_site)
        if not articles_data:
            results.append({
                "url": site,
                "error": "Не найдено статей (проверьте RSS или HTML-ссылку)",
                "articles": []
            })
            continue
        articles_result = []
        total_counts = defaultdict(int)
        for art in articles_data:
            text = art['text']
            preview = art['preview']
            logger.info(f"Article URL: {art['url']}, text length: {len(text)}")
            logger.info(f"Preview: {preview}")
            proc = await asyncio.create_subprocess_exec(
                'python3', CLASSIFIER_PATH,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate(input=text.encode())
            output = stdout.decode()
            logger.info(f"Classifier stdout: {output}")
            if stderr:
                logger.error(f"Classifier stderr: {stderr}")
            dist = {}
            for line in output.split('\n'):
                if ':' in line and '%' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        cat = parts[0].strip()
                        try:
                            percent = float(parts[1].replace('%', '').strip())
                            dist[cat] = percent
                            total_counts[cat] += percent
                        except:
                            pass
            articles_result.append({
                "url": art['url'],
                "preview": preview,
                "distribution": dist
            })
        if total_counts:
            total = sum(total_counts.values())
            norm = {cat: round(val / total * 100, 1) for cat, val in total_counts.items()} if total > 0 else {}
        else:
            norm = {}
        results.append({
            "url": site,
            "articles_parsed": len(articles_data),
            "distribution": norm,
            "articles": articles_result
        })
    return {"results": results}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)