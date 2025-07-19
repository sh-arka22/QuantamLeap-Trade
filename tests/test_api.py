# File: tests/test_api.py
import io
import pytest
from backend.api import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_csv_files(client):
    rv = client.get('/csv_files')
    assert rv.status_code == 200
    data = rv.get_json()
    assert isinstance(data, list)

def test_upload_csv_no_file(client):
    rv = client.post('/upload_csv', data={})
    assert rv.status_code == 400

def test_train_model_no_data(client):
    rv = client.post('/train_model', json={})
    assert rv.status_code == 200
    data = rv.get_json()
    assert "message" in data

def test_get_strategy_performance(client):
    rv = client.get('/strategy_performance')
    assert rv.status_code == 200
    data = rv.get_json()
    assert isinstance(data, list)
    assert all('name' in s and 'profit' in s and 'win_rate' in s for s in data)

def test_get_portfolio(client):
    rv = client.get('/portfolio')
    assert rv.status_code == 200
    data = rv.get_json()
    assert "portfolio_value" in data and "total_profit" in data and "win_rate" in data

def test_predict_invalid(client):
    rv = client.post('/predict', json={})
    assert rv.status_code == 200
    data = rv.get_json()
    assert isinstance(data, dict) or isinstance(data, list)

def test_model_status(client):
    rv = client.get('/model_status')
    assert rv.status_code == 200
    data = rv.get_json()
    assert "status" in data and "accuracy" in data and "mae" in data
