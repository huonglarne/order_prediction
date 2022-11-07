import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.constants import NON_FEATURES
from src.onnx_utils import sklearn_to_onnx

from src.utils import rmse
from src.preprocess_utils import get_past_features, preprocess_data_order

order_data = pd.read_csv("data/data_order.csv")

weekly_sales_data = preprocess_data_order(order_data, "2022-04-30")
weekly_sales_data.to_csv("data/weekly_sales_data.csv", index=False)

features = get_past_features(weekly_sales_data, num_past_weeks=2)
features.to_csv("data/features.csv", index=False)

start_week = (
    int(features.min()["week"]) + 1
)  # each week must have at least 1 week before it to predict
end_week = int(features.max()["week"])
num_weeks = end_week - start_week
train_val_week_split = start_week + round(num_weeks * 0.8)


train = features[features["week"] <= train_val_week_split]
val = features[features["week"] > train_val_week_split]

train_features = train.drop(NON_FEATURES, axis=1).values
train_sales_gt = train["sales"].values

val_features = val.drop(NON_FEATURES, axis=1).values
val_sales_gt = val["sales"].values

# train
model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0)
model.fit(train_features, np.log1p(train_sales_gt))

# val
val_sales_pred = np.expm1(model.predict(val_features))
error = rmse(val_sales_gt, val_sales_pred)
print("Error: %.5f" % error)

sklearn_to_onnx(model, val_features[:1], "model.onnx")
