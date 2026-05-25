from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import subprocess
import httpx
from bs4 import BeautifulSoup
import asyncio
import re
from collections import defaultdict
from newspaper import Article
import feedparser

app = FastAPI()

# CORS
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

@app.post("/classify")
async def classify(request: TextRequest):
    try:
        result = subprocess.run(
            ['python3', '/opt/ai-agent/classifier.py'],
            input=request.text,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {"result": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_text_from_html_with_preview(html: str, url: str) -> tuple:
    """Возвращает (текст, preview)"""
    try:
        article = Article(url)
        article.download(input_html=html)
        article.parse()
        text = article.text
        if not text or len(text) < 200:
            # fallback на BeautifulSoup
            soup = BeautifulSoup(html, 'lxml')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            article_tag = soup.find('article')
            if article_tag:
                text = article_tag.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
        preview = ' '.join(text.split()[:10]) + '...' if text else 'Нет текста'
        return text[:10000], preview
    except:
        return "", "Ошибка извлечения"

async def fetch_article(client, url: str):
    try:
        resp = await client.get(url, timeout=15.0, follow_redirects=True)
        if resp.status_code == 200:
            text, preview = extract_text_from_html_with_preview(resp.text, url)
            if text and len(text) > 200:
                return {'text': text, 'preview': preview, 'url': url}
    except:
        pass
    return None

async def fetch_rss_feed(feed_url: str, max_items: int):
    try:
        feed = feedparser.parse(feed_url)
        entries = feed.entries[:max_items]
        articles = []
        for entry in entries:
            link = entry.get('link')
            if link:
                # Попробуем загрузить статью
                async with httpx.AsyncClient() as client:
                    try:
                        resp = await client.get(link, timeout=15.0, follow_redirects=True)
                        if resp.status_code == 200:
                            text, preview = extract_text_from_html_with_preview(resp.text, link)
                            if text and len(text) > 200:
                                articles.append({'text': text, 'preview': preview, 'url': link})
                    except:
                        pass
        return articles
    except:
        return []

async def fetch_site_articles(site_url: str, max_articles: int):
    # Сначала пробуем найти RSS
    rss_candidates = [f"{site_url}/rss", f"{site_url}/feed", f"{site_url}/rss.xml", f"{site_url}/feed.xml"]
    for rss_url in rss_candidates:
        articles = await fetch_rss_feed(rss_url, max_articles)
        if articles:
            return articles
    # Если RSS нет, собираем ссылки с главной страницы
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(site_url, timeout=10.0, follow_redirects=True)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, 'lxml')
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('/'):
                    href = site_url.rstrip('/') + href
                if href.startswith('http') and site_url.split('/')[2] in href:
                    # Расширенный фильтр
                    if any(key in href.lower() for key in ['news', 'article', 'post', 'story', '202', '2025', '2026', '/p/', '/entry']):
                        links.append(href)
            unique_links = list(dict.fromkeys(links))[:max_articles * 2]  # запас
            # Загружаем статьи
            tasks = [fetch_article(client, link) for link in unique_links]
            results = await asyncio.gather(*tasks)
            articles = [r for r in results if r is not None]
            return articles[:max_articles]
        except:
            return []

@app.post("/analyze_sites")
async def analyze_sites(request: SitesRequest):
    results = []
    for site in request.urls:
        articles_data = await fetch_site_articles(site, request.max_articles_per_site)
        if not articles_data:
            results.append({
                "url": site,
                "error": "Не удалось извлечь тексты статей",
                "articles": []
            })
            continue
        
        # Для каждой статьи сохраняем preview и результат классификации
        articles_result = []
        total_counts = defaultdict(int)
        for art in articles_data:
            text = art['text']
            preview = art['preview']
            # Классификация
            proc = await asyncio.create_subprocess_exec(
                'python3', '/opt/ai-agent/classifier.py',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate(input=text.encode())
            output = stdout.decode()
            # Парсим проценты
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
        
        # Нормализуем общее распределение по сайту
        if total_counts:
            total = sum(total_counts.values())
            if total > 0:
                norm = {cat: round(val / total * 100, 1) for cat, val in total_counts.items()}
            else:
                norm = {}
        else:
            norm = {}
        
        results.append({
            "url": site,
            "articles_parsed": len(articles_data),
            "distribution": norm,
            "articles": articles_result   # детали по каждой статье
        })
    return {"results": results}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)