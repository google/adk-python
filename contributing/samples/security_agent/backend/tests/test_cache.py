from fastapi.testclient import TestClient
from backend.main import app
import time

client = TestClient(app)

def test_cache():
    response = client.get("/api/v1/sessions/test")
    assert response.status_code == 200
    assert response.json() == {"message": "hello"}