def test_metrics_endpoint(base_url, api_session):
    # Заменяем порт 8000 на 8001 (метрики классификатора) или используем /metrics из prometheus?
    # Удобнее проверить напрямую метрики классификатора.
    metrics_url = base_url.replace(":8000", ":8001") + "/metrics"
    resp = api_session.get(metrics_url)
    assert resp.status_code == 200
    assert "yandexgpt_requests_total" in resp.text

def test_metrics_available(base_url, api_session, metrics_url):
    resp = api_session.get(metrics_url)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain; version=0.0.4"

def test_metrics_contain_key_metrics(base_url, api_session, metrics_url):
    resp = api_session.get(metrics_url)
    assert resp.status_code == 200
    body = resp.text
    # Проверяем наличие кастомных метрик
    assert "yandexgpt_requests_total" in body
    assert "yandexgpt_tokens_used_total" in body
    assert "redis_cache_hits_total" in body