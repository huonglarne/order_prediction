from datetime import datetime, date, timedelta
from typing import Tuple
import pandas as pd
from pandas import DataFrame

from src.constants import NUM_PAST_WEEKS, RELEVANT_COLUMNS


def get_past_features(
    weekly_sales_data: DataFrame, num_past_weeks=NUM_PAST_WEEKS
) -> DataFrame:
    """Get past weeks' sales and sales difference as features.

    Args:
        weekly_sales_data (DataFrame): Weekly sales data.
        num_past_weeks (int, optional): Number of previous weeks to consider.

    Returns:
        DataFrame: Past features for prediction model.
    """
    assert set(weekly_sales_data.columns) == {"product", "week", "sales"}
    for i in range(num_past_weeks):
        shift_num = i + 1

        sale_col_name = f"last-{shift_num}_week_sales"
        diff_col_name = f"last-{shift_num}_week_diff"

        weekly_sales_data[sale_col_name] = weekly_sales_data.groupby(["product"])[
            "sales"
        ].shift(shift_num)
        weekly_sales_data[diff_col_name] = weekly_sales_data.groupby(["product"])[
            sale_col_name
        ].diff()

        weekly_sales_data = weekly_sales_data.dropna()

    return weekly_sales_data


def preprocess_order_data(
    order_data: DataFrame, start_date: str = None, end_date: str = None
) -> DataFrame:
    """Preprocess the order data to weekly sales data over a period of all products in database.

    Args:
        order_data (DataFrame): Order data exported from database.
        start_date (str, optional): Start of period to take data from. Format: YYYY-MM-DD.
        If not provided, the earliest date in the DB will be used.
        Must be be accompanied by end_date.
        end_date (str, optional): End of period to take data from. Format: YYYY-MM-DD.

    Returns:
        DataFrame: Weekly sales of all products.
    """
    all_columns = set(order_data.columns)
    assert RELEVANT_COLUMNS.issubset(all_columns)
    order_data = order_data.drop(all_columns - RELEVANT_COLUMNS, axis=1)

    if start_date is None or end_date is None:
        start_date = order_data[order_data["QUANTITY"] > 0]["CHECKOUT_DATE"].min()
        end_date = order_data[order_data["QUANTITY"] > 0]["CHECKOUT_DATE"].max()
        start_date, end_date = _get_period_start_end(start_date, end_date)

    product_id_list = order_data["PRODUCT_ID"].unique()
    weekly_sales_data = None

    for prod_id in product_id_list:
        product_orders = order_data[order_data["PRODUCT_ID"] == prod_id]
        product_weekly_sales = _preprocess_to_weekly_sales(
            product_orders, start_date, end_date
        )

        if weekly_sales_data is not None:
            weekly_sales_data = pd.concat(
                [weekly_sales_data, product_weekly_sales], ignore_index=True
            )
        else:
            weekly_sales_data = product_weekly_sales

    weekly_sales_data = weekly_sales_data.melt(
        id_vars=["PRODUCT_ID"], var_name="week", value_name="sales"
    )

    weekly_sales_data.rename(columns={"PRODUCT_ID": "product"}, inplace=True)
    return weekly_sales_data


def _preprocess_to_weekly_sales(
    product_orders: DataFrame, start_date: str, end_date: str
) -> DataFrame:
    """Preprocess the daily orders to weekly sales data over a period of a single product.
    Args:
        product_orders (DataFrame): Product orders data.
        start_date (str): Start date. Format: YYYY-MM-DD
        end_date (str): End date. Format: YYYY-MM-DD
    Returns:
        DataFrame: Weekly sales of the product.
    """
    assert set(product_orders.columns) == RELEVANT_COLUMNS
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


def _get_period_start_end(start_date: str, end_date: str) -> Tuple[str, str]:
    """Get the start and end date of the period.
    First day of the first month and last day of the last month.

    Args:
        start_date (str): Earilest day recorded in DB. Format: YYYY-MM-DD
        end_date (str): Latest day recorded in DB. Format: YYYY-MM-DD

    Returns:
        Tuple[str, str]: _description_
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    start_date = str(start_date.replace(day=1).date())

    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    end_date = date(
        end_date.year + int(end_date.month / 12), (end_date.month % 12) + 1, 1
    ) - timedelta(
        days=1
    )  # get last day of the month
    end_date = str(end_date)

    return start_date, end_date
