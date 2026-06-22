import pytest
    
def test_health(base_url, api_session):
    resp = api_session.get(f"{base_url}/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}