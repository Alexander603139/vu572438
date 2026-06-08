def test_analyze_sites_single(base_url, api_session):
    urls = ["https://meduza.io"]
    resp = api_session.post(f"{base_url}/analyze_sites", json={"urls": urls, "max_articles_per_site": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    result = data["results"][0]
    assert result["url"] == urls[0]
    assert result["articles_parsed"] >= 1
    assert "avg_confidence" in result
    assert isinstance(result["avg_confidence"], float)