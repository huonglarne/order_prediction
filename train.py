import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.utils import rmse
from src.preprocess_utils import get_past_features, preprocess_data_order

order_data = pd.read_csv("data/data_order.csv")
weekly_sales_data = preprocess_data_order(order_data, "2022-04-30")
features = get_past_features(weekly_sales_data, num_past_weeks=2)

start_week = (
    int(features.min()["week"]) + 1
)  # each week must have at least 1 week before it to predict
end_week = int(features.max()["week"])
num_weeks = end_week - start_week
train_val_week_split = start_week + round(num_weeks * 0.8) + 1


mean_error = []
for week in range(start_week, train_val_week_split):
    train = features[features["week"] < week]
    val = features[features["week"] == week]

    non_features = ["sales", "product", "week"]
    train_features = train.drop(non_features, axis=1)
    train_sales_gt = train["sales"]

    val_features = val.drop(non_features, axis=1)
    val_sales_gt = val["sales"]

    # train
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0)
    model.fit(train_features, np.log1p(train_sales_gt))

    # val
    val_sales_pred = np.expm1(model.predict(val_features))

    error = rmse(val_sales_gt, val_sales_pred)
    mean_error.append(error)

    print("Week %d - Error %.5f" % (week, error))

print("Mean Error: %.5f" % np.mean(mean_error))

# test
test = features[features["week"] >= train_val_week_split]
test_features = test.drop(non_features, axis=1)
test_sales_gt = test["sales"]

test_sales_pred = np.expm1(model.predict(test_features))
error = rmse(test_sales_gt, test_sales_pred)

print("Test Error: %.5f" % error)
