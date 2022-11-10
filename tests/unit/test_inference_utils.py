import pandas as pd
from src.inference_utils import get_product_nextweek_features
from src.preprocess import preprocess_order_data


def test_get_product_latest_features():
    order_data = pd.read_csv("tests/example_data/data_orders.csv")
    weekly_sales_data = preprocess_order_data(order_data)

    _, features = get_product_nextweek_features(4048, weekly_sales_data)
    assert features.shape == (1, 4)
