import pytest

# Примеры текстов для каждой категории (можно расширить)
TEXTS = {
    "Экономические левые": "Государство должно национализировать заводы и повысить налоги на богатство.",
    "Экономические правые": "Необходимо снизить налоги и приватизировать госпредприятия.",
    "Социально-либертарные": "Свобода слова и права ЛГБТ должны быть защищены законом.",
    "Социально-авторитарные": "Традиционные ценности и суверенитет важнее западных свобод."
}

# @pytest.mark.parametrize("expected_label, text", list(TEXTS.items()))
# def test_classify_category(base_url, api_session, expected_label, text):
#     resp = api_session.post(f"{base_url}/classify", json={"text": text})
#     assert resp.status_code == 200
#     data = resp.json()
#     assert "result" in data
#     result = data["result"]
#     # Ищем строку вида "  Экономические левые: 100.0%"
#     assert f"{expected_label}: 100" in result or f"{expected_label}: 100.0" in result

@pytest.mark.parametrize("expected_label, text", list(TEXTS.items()))
def test_classify_category(base_url, api_session, expected_label, text):
    resp = api_session.post(f"{base_url}/classify", json={"text": text})
    assert resp.status_code == 200, f"Ошибка: {resp.text}"
    data = resp.json()
    assert "result" in data, "Нет поля result"
    result = data["result"]
    
    # Ищем строку с категорией и процентом
    lines = [line.strip() for line in result.split("\n") if line.strip()]
    # Фильтруем строки, которые содержат "Уверенность" или "Результат"
    category_lines = [line for line in lines if any(cat in line for cat in TEXTS.keys())]
    # Проверяем, что нужная категория есть и процент близок к 100
    found = False
    for line in category_lines:
        if expected_label in line:
            # Ищем число после ':'
            parts = line.split(':')
            if len(parts) == 2:
                percent_str = parts[1].strip().replace('%', '').replace(',', '.')
                try:
                    percent = float(percent_str)
                    if 99.0 <= percent <= 100.0:
                        found = True
                        break
                except ValueError:
                    pass
    assert found, f"Категория '{expected_label}' не найдена с процентом 100% в ответе: {result}"