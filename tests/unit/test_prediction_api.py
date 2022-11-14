import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.prediction_api import app
from src.inference_utils import get_product_nextweek_features, postprocess_prediction

client = TestClient(app)


# @pytest.mark.integration
def test_predict_product_sale():
    response = client.get("/predict/4048")
    assert response.status_code == 200

    result_dict = response.json()
    assert result_dict["week start"] == "2022-06-06"
    assert result_dict["sales"] > 0
