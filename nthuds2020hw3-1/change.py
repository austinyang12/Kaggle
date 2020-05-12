import pandas as pd
import numpy as np

df_myAns = pd.read_csv("myAns.csv")

print(df_myAns.shape[0])

df_myAns[np.isnan(df_myAns)] = 0

for i in range(df_myAns.shape[0]):
    if df_myAns.loc[i,'RainToday'] == 2:
        df_myAns.loc[i, 'RainToday'] = 1

df_myAns = pd.DataFrame(df_myAns.astype(int), columns = ['RainToday'])
df_myAns.to_csv('myAns2.csv', index_label = 'ID')

print('done')