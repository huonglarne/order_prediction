import pandas as pd
import numpy as np
from src.constants import NON_FEATURES, NUM_PAST_WEEKS
from src.onnx_utils import get_onnx_session, onnx_infere_single_inp_outp

from src.preprocess import get_past_features

weekly_sales_data = pd.read_csv("data/weekly_sales_data.csv")

product_id = 4048

# Get the data for the product with id 4048
prod_sales = weekly_sales_data[weekly_sales_data["product"] == product_id]

prod_sales.loc[len(prod_sales.index)] = {
    "product": product_id,
    "week": prod_sales["week"].max() + 1,
    "sales": 0,
}

num_past_weeks = NUM_PAST_WEEKS
features = get_past_features(
    prod_sales[-(num_past_weeks * 2 + 2) :], num_past_weeks=num_past_weeks
)

features = features.drop(NON_FEATURES, axis=1).values

session = get_onnx_session("checkpoints/model.onnx")
result = onnx_infere_single_inp_outp(session, features)

result = result[:, 0]
result = np.expm1(result)

print(result)
