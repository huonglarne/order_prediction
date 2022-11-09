from calendar import monthrange
from datetime import datetime
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
    assert set(weekly_sales_data.columns) == {"product", "week_start", "sales"}
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
    data_order: DataFrame, start_date: str = None, end_date: str = None
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
    all_columns = set(data_order.columns)
    assert RELEVANT_COLUMNS.issubset(all_columns)
    data_order = data_order.drop(all_columns - RELEVANT_COLUMNS, axis=1)

    data_order["CHECKOUT_DATE"] = pd.to_datetime(
        data_order["CHECKOUT_DATE"], format="%Y-%m-%d"
    )

    data_order = (
        data_order.groupby(
            [pd.Grouper(key="CHECKOUT_DATE", freq="W-MON"), "PRODUCT_ID"]
        )
        .sum()
    )

    data_order = data_order.pivot_table(values=['QUANTITY'], index=['PRODUCT_ID'], columns=['CHECKOUT_DATE'], fill_value=0)
    data_order = data_order.droplevel(0, axis=1).reset_index()
    data_order.columns.name = None

    data_order = data_order.melt(
        id_vars=["PRODUCT_ID"], var_name="week_start", value_name="sales"
    )

    data_order.rename(columns={"PRODUCT_ID": "product"}, inplace=True)
    return data_order


# start_date = start_date or _get_first_day_of_month(
#         order_data[order_data["QUANTITY"] > 0]["CHECKOUT_DATE"].min()
#     )
#     end_date = end_date or _get_last_day_of_month(
#         order_data[order_data["QUANTITY"] > 0]["CHECKOUT_DATE"].max()
#     )



def _get_first_day_of_month(date: str) -> str:
    start_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = str(start_date.replace(day=1).date())
    return start_date


def _get_last_day_of_month(date: str) -> str:
    start_date = datetime.strptime(date, "%Y-%m-%d")
    end_date = str(
        start_date.replace(day=monthrange(start_date.year, start_date.month)[1]).date()
    )
    return end_date
