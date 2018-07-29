import pandas as pd
import datetime as time
import numpy as np

#read csv file
#userlogs = pd.read_csv('user_logs.csv')
chunksize = 10 ** 8
for userlogs in pd.read_csv('user_logs.csv', chunksize=chunksize):
    #get daily usage suitability (percentage of songs more than 98.5)
    userlogs['usage_suitability'] = (userlogs['num_100'] + userlogs['num_985'])/(userlogs['num_100'] + userlogs['num_985'] + userlogs['num_75'] + userlogs['num_50'] + userlogs['num_25'])
    test1 = pd.DataFrame({'usage_above_985' : userlogs.groupby(['msno'])['usage_suitability'].mean()}).reset_index()

    #get number of unique days he used the app
    test2 = pd.DataFrame({'total_days' : userlogs.groupby(['msno'])['date'].count()}).reset_index()

    #get mean of total_secs he used the app
    test3 = pd.DataFrame({'total_secs_mean' : userlogs.groupby(['msno'])['total_secs'].mean()}).reset_index()

    #get mean num_unq
    test4 = pd.DataFrame({'num_unq_mean' : userlogs.groupby(['msno'])['num_unq'].mean()}).reset_index()

    #get cumulative of total_secs he used the app
    test5 = pd.DataFrame({'total_secs_sum' : userlogs.groupby(['msno'])['total_secs'].sum()}).reset_index()

    result = test1.merge(test2, left_on='msno', right_on='msno', how='outer')
    result = result.merge(test3, left_on='msno', right_on='msno', how='outer')
    result = result.merge(test4, left_on='msno', right_on='msno', how='outer')
    result = result.merge(test5, left_on='msno', right_on='msno', how='outer')

    #print first 5 rows
    print(result.head(5))

    result.to_csv('processed_userlogs.csv',mode='a',index=False)
