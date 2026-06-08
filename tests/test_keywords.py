def test_keywords_non_empty(base_url, api_session):
    text = "Государство должно национализировать заводы для суверенитета и традиционных ценностей."
    resp = api_session.post(f"{base_url}/keywords", json={"text": text, "max_per_category": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "keywords" in data
    keywords = data["keywords"]
    assert isinstance(keywords, dict)
    # Проверяем, что значения — списки (могут быть пустыми)
    assert all(isinstance(v, list) for v in keywords.values())

def test_keywords_without_text(base_url, api_session):
    resp = api_session.post(f"{base_url}/keywords", json={"text": "", "max_per_category": 3})
    # API возвращает 200 с пустыми ключевыми словами — это нормально
    assert resp.status_code == 200
    data = resp.json()
    assert "keywords" in data
    assert all(len(v) == 0 for v in data["keywords"].values())