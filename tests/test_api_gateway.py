from fastapi.testclient import TestClient
from src.prediction_api import app

client = TestClient(app)


def test_predict_product_sale():
    response = client.get("/predict/4048")
    assert response.status_code == 200

    result_dict = response.json()
    assert result_dict["week start"] == "2022-05-09"
    assert result_dict["sales"] > 0