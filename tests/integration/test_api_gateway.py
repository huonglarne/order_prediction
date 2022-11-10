import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.prediction_api import app
from src.inference_utils import get_product_nextweek_features, postprocess_prediction

client = TestClient(app)


@pytest.mark.integration
def test_predict_product_sale():
    response = client.get("/predict/4048")
    assert response.status_code == 200

    result_dict = response.json()
    assert result_dict["week start"] == "2022-06-06"
    assert result_dict["sales"] > 0


def test_predict_product_sale():
    weekly_sales_data = pd.read_csv("data/weekly_sales_data.csv")
    product_id_list = [4048, 1168]
    features = [
        get_product_nextweek_features(product_id, weekly_sales_data)[1]
        for product_id in product_id_list
    ]

    features = np.concatenate(features).tolist()

    response = client.post(
        "/batch_inference",
        json={"features": features},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200

    prediction = response.json()["prediction"]
    assert len(prediction) == len(features)
    assert prediction[0] > 0
