import pandas as pd
import numpy as np
from src.constants import NON_FEATURES
from src.onnx_utils import get_onnx_session, onnx_inference_single_input

from src.preprocess_utils import get_past_features

weekly_sales_data = pd.read_csv("data/weekly_sales_data.csv")

prod_id = 4048

# Get the data for the product with id 4048
prod_sales = weekly_sales_data[weekly_sales_data["product"] == prod_id]

prod_sales.loc[len(prod_sales.index)] = {"product": prod_id, "week": prod_sales["week"].max()+1, "sales": 0}

num_past_weeks=2
features = get_past_features(prod_sales[-(num_past_weeks*2+2):], num_past_weeks=num_past_weeks)

features = features.drop(NON_FEATURES, axis=1).values

session = get_onnx_session("model.onnx")
result = onnx_inference_single_input(session, features)

result = result[:, 0]
result = np.expm1(result)

print(result)