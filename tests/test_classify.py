import pytest

# Примеры текстов для каждой категории (можно расширить)
TEXTS = {
    "Экономические левые": "Государство должно национализировать заводы и повысить налоги на богатство.",
    "Экономические правые": "Необходимо снизить налоги и приватизировать государственные предприятия.",
    "Социально-либертарные": "Свобода слова и права ЛГБТ должны быть защищены законом.",
    "Социально-авторитарные": "Традиционные ценности и суверенитет важнее западных свобод."
}

@pytest.mark.parametrize("expected_label, text", list(TEXTS.items()))
def test_classify_category(base_url, api_session, expected_label, text):
    resp = api_session.post(f"{base_url}/classify", json={"text": text})
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    result = data["result"]
    # Ищем строку вида "  Экономические левые: 100.0%"
    assert f"{expected_label}: 100" in result or f"{expected_label}: 100.0" in result