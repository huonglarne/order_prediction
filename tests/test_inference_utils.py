import pandas as pd
import pickle
import numpy as np
from fastapi import FastAPI

from src.onnx_utils import (
    get_onnx_session,
    onnx_infere_single_inp_outp,
    sklearn_to_onnx,
)
from src.inference_utils import get_product_nextweek_features, postprocess_prediction
from src.preprocess import preprocess_order_data


def test_get_product_latest_features():
    order_data = pd.read_csv("tests/test_data/data_orders.csv")
    weekly_sales_data = preprocess_order_data(order_data)
    
    features = get_product_nextweek_features(4048, weekly_sales_data)
    assert features.shape == (1, 4)
