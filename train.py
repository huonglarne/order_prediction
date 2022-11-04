import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

data = pd.read_csv('data/sales_data.csv')

features = data.copy()

past_weeks = 2

for i in range(past_weeks):
    shift_num = i + 1

    sale_column_name = f'last-{shift_num}_week_sales'
    diff_column_name = f'last-{shift_num}_week_diff'
    features[sale_column_name] = features.groupby(['product'])['sales'].shift(shift_num)
    features[diff_column_name] = features.groupby(['product'])[sale_column_name].diff()
    
    features = features.dropna()

start_week = int(features.min()['week']) + 1 # each week must have at least 1 week before it to predict
end_week = int(features.max()['week'])
num_weeks = end_week - start_week
train_week_split = start_week + round(num_weeks * 0.8)

def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))

mean_error = []
for week in range(start_week, train_week_split):
    train = features[features['week'] < week]
    val = features[features['week'] == week]
    
    xtr, xts = train.drop(['sales'], axis=1), val.drop(['sales'], axis=1)
    ytr, yts = train['sales'].values, val['sales'].values
    
    mdl = LGBMRegressor(n_estimators=1000, learning_rate=0.01)
    mdl.fit(xtr, np.log1p(ytr))
    
    p = np.expm1(mdl.predict(xts))
    
    error = rmsle(yts, p)
    print('Week %d - Error %.5f' % (week, error))
    mean_error.append(error)

print('Mean Error = %.5f' % np.mean(mean_error))