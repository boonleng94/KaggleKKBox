import pandas as pd
import datetime as time
import numpy as np
from datetime import timedelta

#read csv file
transactions = pd.read_csv('transactions_merged.csv')

#convert dataFrame columns to date time format
transactions['membership_expire_date'] = pd.to_datetime((transactions['membership_expire_date']), format = "%Y%m%d")
transactions['transaction_date'] = pd.to_datetime((transactions['transaction_date']), format = "%Y%m%d")
#get difference in days'
transactions['difference'] = transactions['membership_expire_date'].sub(transactions['transaction_date'], axis=0)
#convert no. of days into float type
transactions['difference'] = transactions['difference'].astype('timedelta64[D]')
transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date']).astype('int64')/np.timedelta64(1,'m').astype('timedelta64[ns]').astype('int64')
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date']).astype('int64')/np.timedelta64(1,'m').astype('timedelta64[ns]').astype('int64')

#Discount
transactions['discount'] = transactions['plan_list_price'] - transactions['actual_amount_paid']

#Renew before/after expiration
transactions['renewB4ExpirationDate'] = np.where((transactions['difference'] >=0) & (transactions['is_cancel'] ==0), '1', '0')
transactions['renewAfterExpirationDate'] = np.where((transactions['difference'] <0) & (transactions['is_cancel'] ==0), '1', '0')
transactions['renewB4ExpirationDate'] = transactions['renewB4ExpirationDate'].astype(str).astype(int)
transactions['renewAfterExpirationDate'] = transactions['renewAfterExpirationDate'].astype(str).astype(int)

#Churn or not
transactions['churnOrNot'] = np.where((transactions['difference'] <-30), '1', '0')

#Trial User
transactions['trialUser'] = np.where(((transactions['plan_list_price'] == 0) & (transactions['actual_amount_paid'] == 0)), '1', '0')
transactions['trialUser'] = transactions['trialUser'].astype(str).astype(int)


#numOfCanceInDecimal
group_df = transactions.groupby('msno')['is_cancel'].agg(['sum','count'])
group_df['is_cancel_rate'] = group_df['sum'] / group_df['count']
group_df.drop(['sum','count'], axis=1, inplace = True)
transactions = transactions.merge(group_df, right_index = True, left_on='msno', right_on='msno', how='outer')

#numOfNoAutoRenewsInDecimal
group_df = transactions.groupby('msno')['is_auto_renew'].agg(['sum','count'])
group_df['is_auto_renew_rate'] = group_df['sum'] / group_df['count']
group_df.drop(['sum','count'], axis=1, inplace = True)
transactions = transactions.merge(group_df, right_index = True, left_on='msno', right_on='msno', how='outer')

#numOfIsDiscountsInDecimal
# filter the data to discount > 0
temp1 = transactions[transactions['discount'] > 0]
# count number of discount per user
temp1 = pd.DataFrame({'numOfDiscounts': temp1.groupby(['msno'])['discount'].count()}).reset_index()
# count number of transactions per user
temp2 = pd.DataFrame({'numOfTxn': transactions.groupby(['msno'])['msno'].count()}).reset_index()
# merge 2 dataframes
temp1 = temp1.merge(temp2, left_on='msno', right_on='msno', how='outer')
# calculate the percentage of the number of discount
temp1['numOfDiscountsInDecimal'] = temp1['numOfDiscounts'] / temp1['numOfTxn']
# drop the numOfDiscounts and numOfTxn columns
temp1.drop(['numOfDiscounts'], axis=1, inplace = True)
# merge 2 dataframes
transactions = transactions.merge(temp1, left_on='msno', right_on='msno', how='outer')
# fill empty columns with 0
transactions.numOfDiscountsInDecimal.fillna(value=0, inplace=True)

transactionsChurn = transactions[['msno', 'churnOrNot']]

g = transactionsChurn.groupby(['msno']).cumcount()
df = transactionsChurn.set_index(['msno', g]).unstack()
df.columns = ['{}{}'.format(i, j+1) for i, j in df.columns]
df = df.reset_index()
transactionsChurn = df.fillna('-1')
transactionsChurn = transactionsChurn[['msno','churnOrNot1', 'churnOrNot2', 'churnOrNot3', 'churnOrNot4', 'churnOrNot5']]

transactionsChurn['churnOrNot1'] = transactionsChurn['churnOrNot1'].astype(str).astype(int)
transactionsChurn['churnOrNot2'] = transactionsChurn['churnOrNot2'].astype(str).astype(int)
transactionsChurn['churnOrNot3'] = transactionsChurn['churnOrNot3'].astype(str).astype(int)
transactionsChurn['churnOrNot4'] = transactionsChurn['churnOrNot4'].astype(str).astype(int)
transactionsChurn['churnOrNot5'] = transactionsChurn['churnOrNot5'].astype(str).astype(int)

transactions.drop(['churnOrNot'], axis=1, inplace = True)

transactions = transactions.merge(transactionsChurn, left_on='msno', right_on='msno', how='outer')
print(transactions.head(5))
print(transactions.dtypes)
transactions.to_csv('processed_transactions.csv', index=False)
