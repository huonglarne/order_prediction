import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("data/sales_data.csv")

features = data.copy()

past_weeks = 2

for i in range(past_weeks):
    shift_num = i + 1

    sale_column_name = f"last-{shift_num}_week_sales"
    diff_column_name = f"last-{shift_num}_week_diff"
    features[sale_column_name] = features.groupby(["product"])["sales"].shift(shift_num)
    features[diff_column_name] = features.groupby(["product"])[sale_column_name].diff()

    features = features.dropna()

start_week = (
    int(features.min()["week"]) + 1
)  # each week must have at least 1 week before it to predict
end_week = int(features.max()["week"])
num_weeks = end_week - start_week
train_val_week_split = start_week + round(num_weeks * 0.8) + 1


def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))


mean_error = []
for week in range(start_week, train_val_week_split):
    train = features[features["week"] < week]
    val = features[features["week"] == week]

    non_features = ["sales", "product", "week"]
    train_features = train.drop(non_features, axis=1)
    train_sales_gt = train["sales"]

    val_features = val.drop(non_features, axis=1)
    val_sales_gt = val["sales"]

    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0)
    model.fit(train_features, np.log1p(train_sales_gt))

    val_sales_pred = np.expm1(model.predict(val_features))

    error = rmsle(val_sales_gt, val_sales_pred)
    print("Week %d - Error %.5f" % (week, error))
    mean_error.append(error)

print("Mean Error: %.5f" % np.mean(mean_error))


test = features[features["week"] >= train_val_week_split]
test_features = test.drop(non_features, axis=1)
test_sales_gt = test["sales"]

test_sales_pred = np.expm1(model.predict(test_features))
error = rmsle(test_sales_gt, test_sales_pred)
print("Test Error: %.5f" % error)
