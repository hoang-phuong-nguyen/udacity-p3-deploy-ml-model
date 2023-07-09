import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    client = TestClient(app)
    return client

def test_say_hello(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"greeting": "Hello World!"}

def test_predict_postive(client):
    resp = client.post(
        "/predict", 
        json={
            "age": 31,
            "workclass": "Private",
            "fnlgt": 45781,
            "education": "Masters",
            "education_num": 14,
            "marital_status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital_gain": 14084,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "United-States"
        }
    )
    assert resp.status_code == 200
    assert resp.json() == {"Salary": ">50K"}

def test_predict_negative(client):
    resp = client.post(
        "/predict", 
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        }
    )
    assert resp.status_code == 200
    assert resp.json() == {"Salary": "<=50K"}
