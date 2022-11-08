from fastapi.testclient import TestClient
from src.prediction_api import pred_api

client = TestClient(pred_api)


def test_predict_product_sale():
    response = client.get("/predict/4048")
    assert response.status_code == 200

    pred = response.json()["sales for next week"]
    assert pred >= 0
