from fastapi.testclient import TestClient

from backend.app.main import app, get_generator


client = TestClient(app)


def test_health_ok() -> None:
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_chat_mock_response_when_model_unavailable() -> None:
    get_generator.cache_clear()
    response = client.post('/api/chat', json={'message': 'Xin chào'})
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        payload = response.json()
        assert 'reply' in payload
        assert payload['backend'] in ('mock', 'transformers')
