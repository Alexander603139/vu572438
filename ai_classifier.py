# ai_classifier.py - рефакторинг с Dependency Injection

import os
import json
import logging
import httpx
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml
import aioredis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Pydantic модели ----------
class TextRequest(BaseModel):
    text: str

class KeywordsRequest(BaseModel):
    text: str
    max_per_category: int = 5


# ---------- Метрики (глобальные, как раньше) ----------
requests_total = Counter('yandexgpt_requests_total', 'Total requests to YandexGPT', ['endpoint'])
tokens_used_total = Counter('yandexgpt_tokens_used_total', 'Tokens consumed', ['endpoint'])
confidence_histogram = Histogram('yandexgpt_confidence', 'Confidence of classification', buckets=[0.5,0.7,0.8,0.9,0.95,0.99])
cache_hits = Counter('redis_cache_hits_total', 'Cache hits')
cache_misses = Counter('redis_cache_misses_total', 'Cache misses')


# ---------- Репозитории ----------
class SampleRepository:
    """Репозиторий для доступа к few-shot примерам (Redis + YAML fallback)"""
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self._samples_key = "samples:all"

    async def get_all_samples(self) -> List[Dict[str, str]]:
        """Возвращает список примеров {text, label} из Redis, если нет — загружает из YAML и сохраняет"""
        samples_json = await self.redis.get(self._samples_key)
        if samples_json:
            return json.loads(samples_json)
        # fallback: загружаем из YAML и сохраняем в Redis
        samples = await self._load_samples_from_yaml()
        await self.redis.set(self._samples_key, json.dumps(samples))
        logger.info(f"Loaded {len(samples)} samples from YAML into Redis")
        return samples

    async def reload_from_yaml(self):
        """Принудительно перезагружает samples из YAML в Redis (используется администратором)"""
        samples = await self._load_samples_from_yaml()
        await self.redis.set(self._samples_key, json.dumps(samples))
        logger.info(f"Reloaded {len(samples)} samples from YAML")
        return samples

    async def _load_samples_from_yaml(self) -> List[Dict[str, str]]:
        samples_dir = Path("/app/samples")
        label_map = {
            "economic_left.yml": "Экономические левые",
            "economic_right.yml": "Экономические правые",
            "social_liberal.yml": "Социально-либертарные",
            "social_authoritarian.yml": "Социально-авторитарные"
        }
        all_samples = []
        for filename, label in label_map.items():
            filepath = samples_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts = yaml.safe_load(f)
                    for text in texts:
                        all_samples.append({"text": text, "label": label})
            else:
                logger.warning(f"Samples file not found: {filepath}")
        return all_samples


class MarkerRepository:
    """Репозиторий для доступа к маркерам категорий (Redis + YAML fallback)"""
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self._markers_key = "markers:all"

    async def get_all_markers(self) -> Dict[str, List[str]]:
        """Возвращает словарь {название_категории: [маркеры]} из Redis или из YAML"""
        markers_json = await self.redis.get(self._markers_key)
        if markers_json:
            return json.loads(markers_json)
        # fallback: загружаем из YAML
        markers = await self._load_markers_from_yaml()
        await self.redis.set(self._markers_key, json.dumps(markers))
        logger.info("Loaded markers from YAML into Redis")
        return markers

    async def _load_markers_from_yaml(self) -> Dict[str, List[str]]:
        markers_path = Path("/app/political_markers.yml")
        if not markers_path.exists():
            logger.warning("political_markers.yml not found")
            return {}
        with open(markers_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        markers = {}
        for cat, cat_data in data['categories'].items():
            markers[cat_data['name']] = cat_data['markers']
        return markers


# ---------- Клиент YandexGPT ----------
class YandexGPTClient:
    """HTTP-клиент для вызова YandexGPT API (классификация и генерация)"""
    def __init__(self, folder_id: str, api_key: str):
        self.folder_id = folder_id
        self.api_key = api_key
        self.classify_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/fewShotTextClassification"
        self.completion_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.headers = {
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json"
        }

    async def classify(self, text: str, samples: List[Dict[str, str]]) -> Dict[str, Any]:
        """Выполняет few-shot классификацию текста"""
        labels = ["Экономические левые", "Экономические правые", "Социально-либертарные", "Социально-авторитарные"]
        payload = {
            "modelUri": f"cls://{self.folder_id}/yandexgpt/rc",
            "taskDescription": "Определи политическую ориентацию текста. Категории: Экономические левые, Экономические правые, Социально-либертарные, Социально-авторитарные.",
            "labels": labels,
            "text": text,
            "samples": samples
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.classify_url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def extract_keywords(self, text: str, max_per_category: int, markers: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Выделяет ключевые фразы по категориям с помощью генеративной модели"""
        categories_prompt = "\n".join([
            f"- {cat}: примеры маркеров: {', '.join(markers[cat][:10])}..."
            for cat in markers
        ])
        prompt = f"""Ты — помощник, который выделяет ключевые политические фразы из текста. У нас есть категории с типичными маркерами:
        {categories_prompt}
        Задача: прочитай текст и для каждой категории выдели до {max_per_category} ключевых слов или коротких фраз (на русском), 
        которые встречаются в тексте и относятся к этой категории. Если для категории ничего не подходит, верни пустой список.
        Ответ должен быть строго в формате JSON, где ключи — названия категорий, значения — списки строк. 
        Пример: {{"Экономические левые": ["национализация", "социальная справедливость"], "Экономические правые": [], ...}}
        Текст для анализа:
        {text}
        """
        payload = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/rc",
            "completionOptions": {
                "stream": False,
                "temperature": 0.2,
                "maxTokens": 800
            },
            "messages": [
                {"role": "system", "text": "Ты — аналитик, выделяющий ключевые фразы по заданным категориям. Отвечай только JSON."},
                {"role": "user", "text": prompt}
            ]
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.completion_url, headers=self.headers, json=payload)
            resp.raise_for_status()
            result = resp.json()
            response_text = result['result']['alternatives'][0]['message']['text']
            # Извлекаем JSON из ответа
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                keywords = json.loads(json_str)
            else:
                keywords = {}
            # Гарантируем наличие всех категорий
            for cat in markers:
                if cat not in keywords:
                    keywords[cat] = []
            return keywords


# ---------- Сервисы (бизнес-логика) ----------
class ClassificationService:
    """Сервис классификации текста с кэшированием и метриками"""
    def __init__(self, yandex_client: YandexGPTClient, sample_repo: SampleRepository, redis_client: aioredis.Redis):
        self.client = yandex_client
        self.sample_repo = sample_repo
        self.redis = redis_client

    async def classify(self, text: str) -> Dict[str, Any]:
        """Классифицирует текст, используя кэш Redis и YandexGPT"""
        # Проверка кэша
        cache_key = f"yandexgpt:{text}"
        cached = await self.redis.get(cache_key)
        if cached:
            cache_hits.inc()
            logger.info("Cache hit")
            return json.loads(cached)
        cache_misses.inc()
        logger.info("Cache miss, calling YandexGPT")
        # Получаем samples
        samples = await self.sample_repo.get_all_samples()
        # Вызываем API
        data = await self.client.classify(text, samples)
        # Обновляем метрики (токены и confidence)
        tokens_raw = data.get("inputTokens", 0)
        try:
            tokens = int(tokens_raw)
        except (ValueError, TypeError):
            tokens = 0
        tokens_used_total.labels(endpoint='/classify').inc(tokens)
        predictions = data.get("predictions", [])
        if predictions:
            max_conf_raw = max(p.get("confidence", 0) for p in predictions)
            try:
                max_conf = float(max_conf_raw)
            except (ValueError, TypeError):
                max_conf = 0.0
            confidence_histogram.observe(max_conf)
        # Сохраняем в кэш на 1 час
        await self.redis.setex(cache_key, 3600, json.dumps(data))
        return data


class KeywordExtractionService:
    """Сервис для выделения ключевых фраз"""
    def __init__(self, yandex_client: YandexGPTClient, marker_repo: MarkerRepository):
        self.client = yandex_client
        self.marker_repo = marker_repo

    async def extract(self, text: str, max_per_category: int) -> Dict[str, List[str]]:
        """Извлекает ключевые фразы из текста по категориям"""
        markers = await self.marker_repo.get_all_markers()
        return await self.client.extract_keywords(text, max_per_category, markers)


# ---------- FastAPI приложение с lifespan и DI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация зависимостей
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_client = await aioredis.from_url(redis_url, decode_responses=True)
    logger.info("Redis connected")

    folder_id = os.environ.get("YC_FOLDER_ID")
    api_key = os.environ.get("YC_API_KEY")
    if not folder_id or not api_key:
        raise RuntimeError("YC_FOLDER_ID and YC_API_KEY must be set")

    yandex_client = YandexGPTClient(folder_id, api_key)
    sample_repo = SampleRepository(redis_client)
    marker_repo = MarkerRepository(redis_client)

    # Загружаем начальные данные в Redis, если их нет
    if not await redis_client.exists("samples:all"):
        await sample_repo.reload_from_yaml()
    if not await redis_client.exists("markers:all"):
        await marker_repo.get_all_markers()

    classification_service = ClassificationService(yandex_client, sample_repo, redis_client)
    keyword_service = KeywordExtractionService(yandex_client, marker_repo)

    # Помещаем сервисы в app.state для доступа в эндпоинтах
    app.state.classification_service = classification_service
    app.state.keyword_service = keyword_service
    app.state.sample_repo = sample_repo
    app.state.redis_client = redis_client

    yield

    # Закрытие Redis при завершении
    await redis_client.close()
    logger.info("Redis connection closed")


app = FastAPI(lifespan=lifespan)


# ---------- Эндпоинты ----------
@app.post("/classify")
async def classify(request: TextRequest):
    """Классифицирует переданный текст и возвращает распределение по категориям с процентами"""
    try:
        service: ClassificationService = app.state.classification_service
        result = await service.classify(request.text)
        predictions = result.get("predictions", [])
        if not predictions:
            raise ValueError("No predictions")
        total = sum(p.get("confidence", 0) for p in predictions)
        if total == 0:
            total = 1.0
        lines = ["Результат классификации:"]
        for p in predictions:
            label = p.get("label")
            conf = p.get("confidence", 0)
            percent = round(conf / total * 100, 1)
            lines.append(f"  {label}: {percent}%")
        max_conf = max(p.get("confidence", 0) for p in predictions)
        lines.append(f"Уверенность ИИ: {max_conf:.2f}")
        result_str = "\n".join(lines)
        return {"result": result_str}
    except Exception as e:
        logger.exception("Error in classify")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_keywords")
async def extract_keywords(request: KeywordsRequest):
    """Выделяет ключевые фразы по категориям из переданного текста"""
    try:
        service: KeywordExtractionService = app.state.keyword_service
        keywords = await service.extract(request.text, request.max_per_category)
        return {"keywords": keywords}
    except Exception as e:
        logger.exception("Error in extract_keywords")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Эндпоинт для сбора метрик Prometheus"""
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")


@app.post("/admin/reload_samples")
async def reload_samples():
    """Административный эндпоинт для перезагрузки few-shot примеров из YAML в Redis"""
    try:
        sample_repo: SampleRepository = app.state.sample_repo
        await sample_repo.reload_from_yaml()
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Error reloading samples")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Проверка работоспособности сервиса"""
    return {"status": "ok"}