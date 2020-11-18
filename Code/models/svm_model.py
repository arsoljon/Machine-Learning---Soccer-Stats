import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#Analyze Combine_data.csv first.
#classes will be [w, d, l] = [0,0,0]
path = "C:/Users/Jonathan/PycharmProjects/Machine-Learning---Soccer-Stats/Data/Combined_data.csv"
df = pd.read_csv(path, sep=',', dtype=str)
df.drop(['Squad'], axis=1, inplace=True)
'''
#Normilize data
data = np.array(df)
data = data.astype(np.float)
s = MinMaxScaler()
s.fit(data)
data = s.transform(data)
#the next line helps organize the data easier
scaled_df = pd.DataFrame(data, columns=df.columns)
'''
#data is 20rows x 13cols
X = np.array(df.drop(['W', 'D', 'L'], axis=1))
Y = np.array(df.loc[:,['W', 'D', 'L']])
#change the greatest value of Y into 1 and make the rest 0
#problem with this method is we lose a layer of accuracy
#used for softmax in lab15
#MAKE ONE COLUMN FOR    w/d/l represented by 1/0/-1
for r in Y:
    if (max(r[0], r[1], r[2]) == r[0]):
        r[0] = 1
        r[1] = 0
        r[2] = 0
    elif (max(r[0], r[1], r[2]) == r[1]):
        r[0] = 0
        r[1] = 0
        r[2] = 0
    elif (max(r[0], r[1], r[2]) == r[2]):
        r[0] = -1
        r[2] = 0
        r[1] = 0

#remove other columns from Y & include 1 col that use 1/0/-1 -> w/d/L
ytemp = []
for r in Y:
    ytemp.append(r[0])
Y = ytemp

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=1)

model = SVC(kernel="linear")
model.fit(train_x, train_y)

pred_y = model.predict(test_x)
acc = accuracy_score(test_y, pred_y)

#save output
filename = "../../Data"
with open('out.txt', 'w') as f:
    print('Filename:', filename, file=f)
print(acc)


'''
#visulaize our data
#will worry about this later.
plt.plot(Y[0,0,], 'bo', markersize = 3)
plt.plot(X, 'ro', markersize = 1)
plt.show()
'''
