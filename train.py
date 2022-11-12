from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
from src.constants import NON_FEATURES, NUM_PAST_WEEKS
from src.inference_utils import postprocess_prediction
from src.onnx_utils import sklearn_to_onnx

from src.utils import rmse
from src.preprocess import get_past_features, preprocess_order_data

# Preprocess data
# Feature engineering
order_data = pd.read_csv("data/data_order.csv")

weekly_sales_data = preprocess_order_data(
    order_data, start_date="2022-02-01", end_date="2022-05-31"
)
weekly_sales_data.to_csv("data/weekly_sales_data.csv", index=False)

features = get_past_features(weekly_sales_data, num_past_weeks=NUM_PAST_WEEKS)


# Split data
start_week = features.min()["week_start"] + timedelta(
    days=7
)  # each week must have at least 1 week before it to predict
end_week = features.max()["week_start"]
num_weeks = (end_week - start_week).days / 7
train_split = start_week + timedelta(round(num_weeks * 0.8 * 7))

train = features[features["week_start"] <= train_split]
val = features[features["week_start"] > train_split]

train_features = train.drop(NON_FEATURES, axis=1).values
train_sales_gt = train["sales"].values

val_features = val.drop(NON_FEATURES, axis=1).values
val_sales_gt = val["sales"].values

# Train and validate model
model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0)
model.fit(train_features, np.log1p(train_sales_gt))

val_sales_pred = postprocess_prediction(model.predict(val_features))
error = rmse(val_sales_gt, val_sales_pred)

print("Error: %.5f" % error)

# Checkpoint
dump(model, "chekpoints/model.joblib")
sklearn_to_onnx(model, val_features[:1], "models/prediction_model/1/model.onnx")
