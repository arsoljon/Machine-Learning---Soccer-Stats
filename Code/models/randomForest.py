import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('2018-2019.csv')
# print(df.head())

# drop nonnumeric data
df.drop(['Date'], axis=1, inplace=True)
df.drop(['HomeTeam'], axis=1, inplace=True)
df.drop(['AwayTeam'], axis=1, inplace=True)
df.drop(['Referee'], axis=1, inplace=True)
df.drop(['HTR'], axis=1, inplace=True)

# convert W, D L to numbers
df.FTR[df.FTR == 'H'] = 1
df.FTR[df.FTR == 'D'] = 0
df.FTR[df.FTR == 'A'] = -1
# print(df.head())

# define dependent variable
Y = df['FTR'].values
Y = Y.astype('int')

# define independent variables
X = df.drop(labels=['FTR'], axis=1)

# split data
from sklearn.model_selection import train_test_split

# initialize random forest size and state
# so that it is not different every time
# we run it

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=30)
model.fit(X_train, Y_train)

prediction_test = model.predict(X_test)
#print(prediction_test)

from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))

#print features by importance
feature_list= list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)
