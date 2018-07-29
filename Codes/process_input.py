import gc;
import pandas as pd
import numpy

gc.enable()

# train file from Kaggle
train_v2 = pd.read_csv('train_v2.csv')
# userlogs file from Kaggle after feature preprocessing
train1 = pd.read_csv('processed_userlogs.csv')
# transactions file from Kaggle after feature preprocessing
train2 = pd.read_csv('processed_transactions.csv')
# merging userlogs with transactions
train3 = pd.merge(train1, train2, how='inner', on='msno')
# all features merging with train file
trainv2 = pd.merge(train3, train_v2, how='right', on='msno')

# test file from Kaggle
test = pd.read_csv('test.csv')
# all features merging with test file
testv2 = pd.merge(test, train3, how='left', on='msno')

# members file from Kaggle
members = pd.read_csv('members_v3.csv')
# final training dataset
train = pd.merge(trainv2, members, how='left', on='msno')
# final test dataset
test = pd.merge(testv2, members, how='left', on='msno')
# processing gender to numerical data
gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)
# fill empty data
train = train.fillna(-1)
# fill empty data
test = test.fillna(-1)

test.to_csv('test_input.csv', index=False)
train.to_csv('train_input.csv', index=False)
