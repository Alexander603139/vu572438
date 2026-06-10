import feedparser
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

async def fetch_telegram_posts(username: str, max_posts: int) -> List[Dict]:
    """
    Получает посты из публичного Telegram-канала по RSS.
    username — имя канала (без @ и без t.me/). Пример: 'durov'
    max_posts — максимум постов (от 1 до 20)
    """
    rss_url = f"https://t.me/s/{username}.rss"
    feed = feedparser.parse(rss_url)
    if feed.bozo:
        logger.warning(f"RSS parse error for {username}: {feed.bozo_exception}")
        raise Exception("Канал не найден или недоступен")
    entries = feed.entries[:max_posts]
    posts = []
    for entry in entries:
        # Текст поста может быть в summary, content или description
        text = entry.get('summary', '') or entry.get('description', '') or ''
        if not text:
            content = entry.get('content', [])
            if content:
                text = content[0].get('value', '')
        # Удаляем HTML-теги
        text = re.sub(r'<[^>]+>', '', text).strip()
        if not text:
            continue
        preview = ' '.join(text.split()[:10]) + '...'
        posts.append({
            "text": text[:10000],
            "preview": preview,
            "url": entry.get('link', f"https://t.me/{username}")
        })
    return posts