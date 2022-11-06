import pandas as pd
from src.preprocess_utils import preprocess_data_order, preprocess_to_weekly_sales

def test_preprocess_data_order():
    order_data = pd.read_csv("tests/test_data/data_orders.csv")
    weekly_sales_data = preprocess_data_order(order_data, "2022-04-30")
    assert weekly_sales_data.columns.tolist() == ["product", "week", "sales"]
    assert weekly_sales_data.shape == (18*3, 3)



def test_preprocess_to_weekly_sales():
    product_orders = pd.read_csv("tests/test_data/product_orders.csv")
    product_weekly_sales = preprocess_to_weekly_sales(product_orders, "2022-01-01", "2022-04-30")
    assert product_weekly_sales.iloc[0].tolist() == [
        4048,
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
