
import gc;
import pandas as pd
import numpy
import math
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb
import sklearn

from sklearn import *
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

gc.enable()

train = pd.read_csv('train_input_filtered.csv')
test = pd.read_csv('test_input_filtered.csv')

# get total columns (features)
cols = [c for c in train.columns if c not in ['is_churn','msno']]
print(cols)

# define xgb_score metric for logloss
def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)

# parameters for xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 7,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 3228,
    'silent': True,
    'tree_method': 'exact'
}
# parameters for lgb
lgb_params = {
    'learning_rate': 0.05,
    'application': 'binary',
    'max_depth': 7,
    'num_leaves': 256,
    'verbosity': -1,
    'metric': 'binary_logloss',
    'num_boost_round': 600,
    'early_stopping_rounds': 50
}

#train = train.sample(100000)

# splitting training dataset to train and test evaluate the models
x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.2, random_state=3228)

# catboost
print('cat training')

model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=7, l2_leaf_reg=5, loss_function='Logloss', eval_metric='Logloss')
# training dataset
model = model.fit(x1, y1,eval_set=(x2,y2),logging_level='Silent')

# actual test dataset
cat_pred = model.predict_proba(test[cols])[:,1]

# training-test dataset
cat_valid = model.predict_proba(x2)[:,1]
print('cat valid log loss = {}'.format(log_loss(y2,cat_valid)))

# xgb
print('xgb training')

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

# training dataset
model = xgb.train(xgb_params, xgb.DMatrix(x1, y1), 500,  watchlist, feval=xgb_score, maximize=False, verbose_eval=100, early_stopping_rounds=50)

# actual test dataset
xgb_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

# training-test dataset
xgb_valid = model.predict(xgb.DMatrix(x2))
print('xgb valid log loss = {}'.format(log_loss(y2,xgb_valid)))

# lgbm
print('lgb training')
d_train = lgb.Dataset(x1, label=y1)
d_valid = lgb.Dataset(x2, label=y2)
watchlist = [d_train, d_valid]

# training dataset
model = lgb.train(lgb_params, train_set=d_train, valid_sets=watchlist, verbose_eval=100)

# feature importance of lgb
ax = lgb.plot_importance(model)
plt.tight_layout()
plt.savefig('feature_importance_graph.png')

# actual test dataset
lgb_pred = model.predict(test[cols])

# training-test dataset
lgb_valid = model.predict(x2)
print('lgb valid log loss = {}'.format(log_loss(y2,lgb_valid)))

# averaging
merged_pred = (cat_pred + xgb_pred + lgb_pred) / 3

test['is_churn'] = cat_pred.clip(0.+1e-15, 1-1e-15)
test = pd.DataFrame({'is_churn' : test.groupby(['msno'])['is_churn'].mean()}).reset_index()
test[['msno','is_churn']].to_csv('cat.csv', index=False)

test['is_churn'] = xgb_pred.clip(0.+1e-15, 1-1e-15)
test = pd.DataFrame({'is_churn' : test.groupby(['msno'])['is_churn'].mean()}).reset_index()
test[['msno','is_churn']].to_csv('xgb.csv', index=False)

test['is_churn'] = lgb_pred.clip(0.+1e-15, 1-1e-15)
test = pd.DataFrame({'is_churn' : test.groupby(['msno'])['is_churn'].mean()}).reset_index()
test[['msno','is_churn']].to_csv('lgb.csv', index=False)

test['is_churn'] = merged_pred.clip(0.+1e-15, 1-1e-15)
test = pd.DataFrame({'is_churn' : test.groupby(['msno'])['is_churn'].mean()}).reset_index()
test[['msno','is_churn']].to_csv('ensemble.csv', index=False)
