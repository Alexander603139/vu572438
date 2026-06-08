def test_keywords_non_empty(base_url, api_session):
    text = "Государство должно национализировать заводы для суверенитета и традиционных ценностей."
    resp = api_session.post(f"{base_url}/keywords", json={"text": text, "max_per_category": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "keywords" in data
    keywords = data["keywords"]
    assert isinstance(keywords, dict)
    # Хотя бы одна категория должна получить фразу
    total_phrases = sum(len(v) for v in keywords.values())
    assert total_phrases > 0

def test_keywords_without_text(base_url, api_session):
    resp = api_session.post(f"{base_url}/keywords", json={"text": "", "max_per_category": 3})
    assert resp.status_code == 500 or resp.status_code == 422  # ожидаем ошибку