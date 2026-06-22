import pytest

class TestClassify:
    def test_health(self, base_url, api_session):
        resp = api_session.get(f"{base_url}/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_classify_left(self, base_url, api_session):
        text = "Государство должно национализировать заводы и повысить налоги."
        resp = api_session.post(f"{base_url}/classify", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        result = data["result"]
        assert "Экономические левые: 100.0%" in result or "Экономические левые: 100%" in result

    def test_classify_right(self, base_url, api_session):
        text = "Необходимо снизить налоги и приватизировать госпредприятия."
        resp = api_session.post(f"{base_url}/classify", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        result = data["result"]
        assert "Экономические правые: 100.0%" in result or "Экономические правые: 100%" in result

    def test_extract_keywords(self, base_url, api_session):
        text = "Государство должно национализировать заводы для суверенитета."
        resp = api_session.post(f"{base_url}/keywords", json={"text": text, "max_per_category": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert "keywords" in data
        keywords = data["keywords"]
        assert isinstance(keywords, dict)
        assert all(isinstance(v, list) for v in keywords.values())

    def test_analyze_sites(self, base_url, api_session):
        urls = ["https://meduza.io"]
        resp = api_session.post(f"{base_url}/analyze_sites", json={"urls": urls, "max_articles_per_site": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        result = data["results"][0]
        assert "articles_parsed" in result
        assert result["articles_parsed"] >= 1
        assert "avg_confidence" in result