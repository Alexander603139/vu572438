import os
import pytest
import requests

@pytest.fixture(scope="session")
def base_url():
    """Базовый URL API (из переменной окружения или localhost)"""
    return os.environ.get("AI_AGENT_URL", "http://localhost:8000")

@pytest.fixture(scope="session")
def api_session():
    """Сессия requests с JSON-заголовком"""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session

@pytest.fixture(scope="session")
def metrics_url(base_url):
    """URL для метрик классификатора (порт 8001)"""
    return base_url.replace(":8000", ":8001") + "/metrics"