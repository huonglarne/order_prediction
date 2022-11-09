from datetime import timedelta
import pandas as pd
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from src.preprocess import get_past_features
from src.constants import NON_FEATURES, NUM_PAST_WEEKS


def get_product_nextweek_features(
    product_id: int,
    weekly_sales_data: DataFrame,
    num_past_weeks: int = NUM_PAST_WEEKS,
) -> ndarray:
    assert set(weekly_sales_data.columns) == {"week_start", "product", "sales"}
    
    if weekly_sales_data["week_start"].dtypes != "datetime64[ns]":
        weekly_sales_data["week_start"] = pd.to_datetime(weekly_sales_data["week_start"])

    prod_sales = weekly_sales_data[
        weekly_sales_data["product"] == product_id
    ]

    next_week = prod_sales["week_start"].max() + timedelta(days=7)
    prod_sales.loc[len(prod_sales.index)] = {
        "product": product_id,
        "week_start": next_week,
        "sales": 0,
    }

    features = get_past_features(
        prod_sales[-(num_past_weeks * 2 + 2) :], num_past_weeks=num_past_weeks
    )

    features = features.drop(NON_FEATURES, axis=1).values
    return next_week, features


def postprocess_prediction(result: ndarray) -> ndarray:
    result = np.expm1(result)
    return result
