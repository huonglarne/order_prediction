import pandas as pd
from src.preprocess import (
    get_past_features,
    preprocess_order_data
)


def test_get_past_features():
    """Docstring"""
    order_data = pd.read_csv("tests/test_data/data_orders.csv")
    weekly_sales_data = preprocess_order_data(order_data, "2022-01-01", "2022-04-30")

    features = get_past_features(weekly_sales_data, num_past_weeks=1)

    assert set(features.columns) == {
        "product",
        "week_start",
        "sales",
        "last-1_week_sales",
        "last-1_week_diff",
    }

    assert features.shape[0] == 48


def test_preprocess_data_order():
    """Docstring"""
    product_orders = pd.read_csv("tests/test_data/product_orders.csv")
    product_weekly_sales_data = preprocess_order_data(product_orders)

    assert set(weekly_sales_data.columns) == {"product", "week_start", "sales"}
    assert product_weekly_sales_data["sales"].tolist() == [
        2,
        8,
        11,
        21,
        7,
        16,
        10,
        18,
        2,
        9,
        11,
        10,
        12,
        10,
        4,
        7,
        5,
        3,
    ]


    order_data = pd.read_csv("tests/test_data/data_orders.csv")
    weekly_sales_data = preprocess_order_data(order_data, start_date="2022-02-11", end_date="2022-04-30")

    assert weekly_sales_data["week_start"].min() == pd.Timestamp("2022-02-14")
    assert weekly_sales_data["week_start"].max() == pd.Timestamp("2022-05-02")