import sklearn as sk
import pandas as pd
import numpy as np
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import datasets

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
#data is 20rows x 13cols
X = np.array(scaled_df.drop(['W', 'D', 'L'], axis=1))
Y = np.array(scaled_df.loc[:,['W', 'D', 'L']])
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)

#visulaize our data
#will worry about this later.
plt.plot(Y[0,0,], 'bo', markersize = 3)
plt.plot(X, 'ro', markersize = 1)
plt.show()