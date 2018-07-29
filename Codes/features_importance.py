
import pandas as pd
import numpy

test = pd.read_csv('test_input.csv')
test = test.drop(['gender', 'registered_via', 'plan_list_price', 'is_auto_renew', 'payment_plan_days', 'renewB4ExpirationDate', 'churnOrNot3', 'is_cancel', 'discount', 'churnOrNot2', 'churnOrNot5', 'churnOrNot4', 'trialUser', 'renewAfterExpirationDate', 'churnOrNot1'], axis=1)
test.to_csv('test_input_filtered.csv', index=False)

train = pd.read_csv('train_input.csv')
train = train.drop(['gender', 'registered_via', 'plan_list_price', 'is_auto_renew', 'payment_plan_days', 'renewB4ExpirationDate', 'churnOrNot3', 'is_cancel', 'discount', 'churnOrNot2', 'churnOrNot5', 'churnOrNot4', 'trialUser', 'renewAfterExpirationDate', 'churnOrNot1'], axis=1)
train.to_csv('train_input_filtered.csv', index=False)
