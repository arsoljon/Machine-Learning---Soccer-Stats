import sklearn as sk
import pandas as pd
import numpy as np
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Analyze Combine_data.csv first.
#classes will be [w, d, l] = [0,0,0]
path = "C:/Users/Jonathan/PycharmProjects/Machine-Learning---Soccer-Stats/Data/Combined_data.csv"
df = pd.read_csv(path, sep=',', dtype=str)
df.drop(['Squad'], axis=1, inplace=True)
#Normilize data
data = np.array(df)
data = data.astype(np.float)
s = MinMaxScaler()
s.fit(data)
data = s.transform(data)
#the next line helps organize the data easier
scaled_df = pd.DataFrame(data, columns=df.columns)
print(scaled_df)
#data is 20rows x 13cols
#make the train & test data
train_data = scaled_df.loc[:14]     #number of rows = 15
test_data = scaled_df.loc[15:]      #number of rows = 5

train_x = scaled_df.drop(['W', 'D', 'L'], axis=1)
train_y = scaled_df.loc[:,['W', 'D', 'L']]
test_x = scaled_df.drop(['W', 'D', 'L'], axis=1)
test_y = scaled_df.loc[:,['W', 'D', 'L']]


