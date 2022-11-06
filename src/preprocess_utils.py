import pandas as pd
from pandas import DataFrame


def preprocess_to_weekly_sales(product_orders: DataFrame, start_date: str, end_date: str) -> DataFrame:
    """Preprocess the daily product orders to weekly sales data over a period.
    Args:
        product_orders (DataFrame): Product orders data.
        start_date (str): Start date. Format: YYYY-MM-DD
        end_date (str): End date. Format: YYYY-MM-DD
    Returns:
        DataFrame: Preprocessed product orders data.
    """
    assert set(product_orders.columns) == {
        "PRODUCT_ID",
        "QUANTITY",
        "CHECKOUT_DATE",
    }
    prod_id = product_orders["PRODUCT_ID"].unique()[0]

    # sort by CHECKOUT_DATE
    product_orders["CHECKOUT_DATE"] = pd.to_datetime(
        product_orders["CHECKOUT_DATE"], format="%Y-%m-%d"
    )

    # sum of sales for each day.
    product_orders = (
        product_orders.groupby(["CHECKOUT_DATE", "PRODUCT_ID"])
        .sum()
        .sort_values(by="CHECKOUT_DATE")
        .reset_index()
    )

    product_orders = _fill_missing_dates(product_orders, start_date, end_date)

    # convert daily to weekly data
    product_orders = (
        product_orders.groupby(
            [pd.Grouper(key="CHECKOUT_DATE", freq="W-MON"), "PRODUCT_ID"]
        )
        .sum()
        .sort_values(by="CHECKOUT_DATE")
    )
    product_orders = product_orders.reset_index().drop(
        ["CHECKOUT_DATE", "PRODUCT_ID"], axis=1
    )

    # turn each week's sale into a feature
    product_orders = product_orders.T.reset_index().drop(["index"], axis=1)
    product_orders.insert(loc=0, column="PRODUCT_ID", value=[prod_id])

    return product_orders


def _fill_missing_dates(
    product_orders: DataFrame, start_date: str, end_date: str
) -> DataFrame:
    """Fill missing dates with 0 sales.
    Args:
        product_orders (DataFrame): Product orders data.
        start_date (str): Start date. Format: YYYY-MM-DD
        end_date (str): End date. Format: YYYY-MM-DD
    Returns:
        DataFrame: Product orders data with missing dates filled.
    """
    prod_id = product_orders["PRODUCT_ID"].unique()[0]

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    fill_values = {"PRODUCT_ID": prod_id, "QUANTITY": 0}
    product_orders = (
        product_orders.set_index("CHECKOUT_DATE")
        .reindex(date_range)
        .fillna(value=fill_values)
        .rename_axis("CHECKOUT_DATE")
        .reset_index()
    )
    return product_orders
