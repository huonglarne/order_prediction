import numpy as np
from numpy import ndarray
from pandas import DataFrame
from src.preprocess import get_past_features
from src.constants import NON_FEATURES, NUM_PAST_WEEKS


def get_product_latest_features(
    product_id: int,
    latest_weekly_sales_data: DataFrame,
    num_past_weeks: int = NUM_PAST_WEEKS,
) -> ndarray:
    prod_sales = latest_weekly_sales_data[
        latest_weekly_sales_data["product"] == product_id
    ]
    prod_sales.loc[len(prod_sales.index)] = {
        "product": product_id,
        "week": prod_sales["week"].max() + 1,
        "sales": 0,
    }

    features = get_past_features(
        prod_sales[-(num_past_weeks * 2 + 2) :], num_past_weeks=num_past_weeks
    )

    features = features.drop(NON_FEATURES, axis=1).values
    return features


def postprocess_prediction(result: ndarray) -> ndarray:
    result = np.expm1(result)
    return result
