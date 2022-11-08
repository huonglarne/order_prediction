import pandas as pd
import pickle
import numpy as np
from fastapi import FastAPI

from src.onnx_utils import (
    get_onnx_session,
    onnx_infere_single_inp_outp,
    sklearn_to_onnx,
)
from src.inference_utils import get_product_latest_features, postprocess_prediction


def test_get_product_latest_features():
    weekly_sales_data = pd.read_csv("data/weekly_sales_data.csv")
    features = get_product_latest_features(4048, weekly_sales_data)
    assert features.shape == (1, 4)
