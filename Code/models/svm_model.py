import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns

#Analyze Combine_data.csv first.
#classes will be [w, d, l] = [0,0,0]
def combined_svm():
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

    #prepocessing
    #data is 20rows x 13cols
    X = np.array(scaled_df.drop(['W', 'D', 'L'], axis=1), dtype=float)
    Y = np.array(scaled_df.loc[:,['W', 'D', 'L']])
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
            r[0] = 2
            r[2] = 0
            r[1] = 0
    #remove other columns from Y & include 1 col that use 1/0/-1 -> w/d/L
    ytemp = []
    for r in Y:
        ytemp.append(r[0])
    Y = ytemp

    #svm model
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=20)
    model = SVC(kernel="linear" )
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print(model)
    print(acc)

    #plot
    data_x = scaled_df.drop(['W', 'D', 'L'], axis=1)   #used for the plot only
    data_y = scaled_df.loc[:, ['W']]                   #used for the plot only
    #For some reasone a for loop would not change the values in data_y. -_-
    #this is not needed.
    data_y.loc[:, 'W'][data_y.loc[:, 'W'] != False] = Y
    #Plotting the Possesion & points
    plt.scatter(data_x.iloc[:, 4], data_x.iloc[:, 9],c = 'coral', s=50, cmap='autumn')
    plt.title('Stats for the Season')
    plt.xlabel('Possession%')
    plt.ylabel('Points earned')


#make svm for matches without betting
def matches_wobo_2019_2020_svm():
    path = "C:/Users/Jonathan/PycharmProjects/Machine-Learning---Soccer-Stats/Data/matches-wobo-2019-2020.csv"
    df = pd.read_csv(path, sep=',', dtype=str)
    df.drop(['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'Referee'], axis=1, inplace=True)
    #y is FTR, Full Time Result. Denoted with a H/D/A   for Home win/ Draw/ Away win
    #Essentially, This will discover if teams win more often when they play as home or away.
    #can try a  differeent y later. Maybe creating separate testing data with only team matches.
    #can add referee id later to include referees
    #change the values for HTR, Half time results.
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'H'] = 1.0
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'D'] = 0.0
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'A'] = 2
    df.loc[:, 'FTR'][df.loc[:, 'FTR'] == 'H'] = 1.0
    df.loc[:, 'FTR'][df.loc[:, 'FTR'] == 'D'] = 0.0
    df.loc[:, 'FTR'][df.loc[:, 'FTR'] == 'A'] = 2.0
    # Normilize data
    data = np.array(df)
    data = data.astype(np.float)
    s = MinMaxScaler()
    s.fit(data)
    data = s.transform(data)
    # the next line helps organize the data easier
    scaled_df = pd.DataFrame(data, columns=df.columns)
    #removed the htr column, halftime results. This increases accuracy.
    X = np.array(scaled_df.drop(['FTR', 'HTR'], axis=1))
    Y = np.array(df.loc[:, ['FTR']])
    Y = Y.ravel().tolist()

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=1)

    model = SVC(kernel="linear", degree=7)
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print(acc)
    '''
    #Plot
    data_x = scaled_df  #used for the plot only
    data_y = scaled_df.loc[:,['FTHG'] ]
    ytemp = []
    for i in range(data_y.index.stop):
        ytemp.append(i + 1)
    #created a count for each of the games
    data_y.loc[:, 'FTHG'][data_y.loc[:, 'FTHG'] != False] = ytemp
    plt.scatter(data_y.iloc[:,0], data_x.iloc[:, 5], c = 'coral', s=50, cmap='autumn')
    plt.title('Match Results')
    plt.xlabel('Single Match')
    plt.ylabel('Halftime points won')
    plt.show()
    plt.scatter(data_y.iloc[:,0], data_x.iloc[:, 2], c = 'blue', s=50, cmap='autumn')
    plt.title('Match Results')
    plt.xlabel('Single Match')
    plt.ylabel('Fulltime points won')
    plt.show()
    '''
    return df

def matches_wobo_2018_2019_svm():
    path = "C:/Users/Jonathan/PycharmProjects/Machine-Learning---Soccer-Stats/Data/matches-wobo-2018-2019.csv"
    df = pd.read_csv(path, sep=',', dtype=str)
    df.drop(['Div', 'Date', 'HomeTeam', 'AwayTeam', 'Referee'], axis=1, inplace=True)
    #Preprocessing
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'H'] = 1.0
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'D'] = 0.0
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'A'] = 2
    df.loc[:, 'FTR'][df.loc[:, 'FTR'] == 'H'] = 1.0
    df.loc[:, 'FTR'][df.loc[:, 'FTR'] == 'D'] = 0.0
    df.loc[:, 'FTR'][df.loc[:, 'FTR'] == 'A'] = 2.0
    # Normilize data
    data = np.array(df)
    data = data.astype(np.float)
    s = MinMaxScaler()
    s.fit(data)
    data = s.transform(data)
    # the next line helps organize the data easier
    scaled_df = pd.DataFrame(data, columns=df.columns)
    # removed the htr column, halftime results. This increases accuracy.
    X = np.array(scaled_df.drop(['FTR', 'HTR'], axis=1))
    Y = np.array(df.loc[:, ['FTR']])
    Y = Y.ravel().tolist()

    #train
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=1)
    model = SVC(kernel="linear", degree=7)
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print(acc)
    return df

def matches_wobo_2017_2018_svm():
    path = "C:/Users/Jonathan/PycharmProjects/Machine-Learning---Soccer-Stats/Data/matches-wobo-2017-2018.csv"
    df = pd.read_csv(path, sep=',', dtype=str)
    df.drop(['Div', 'Date', 'HomeTeam', 'AwayTeam', 'Referee'], axis=1, inplace=True)
    #Preprocessing
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'H'] = 1.0
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'D'] = 0.0
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'A'] = 2
    df.loc[:, 'FTR'][df.loc[:, 'FTR'] == 'H'] = 1.0
    df.loc[:, 'FTR'][df.loc[:, 'FTR'] == 'D'] = 0.0
    df.loc[:, 'FTR'][df.loc[:, 'FTR'] == 'A'] = 2.0
    # Normilize data
    data = np.array(df)
    data = data.astype(np.float)
    s = MinMaxScaler()
    s.fit(data)
    data = s.transform(data)
    # the next line helps organize the data easier
    scaled_df = pd.DataFrame(data, columns=df.columns)
    # removed the htr column, halftime results. This increases accuracy.
    X = np.array(scaled_df.drop(['FTR', 'HTR'], axis=1))
    Y = np.array(df.loc[:, ['FTR']])
    Y = Y.ravel().tolist()

    #train
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=1)
    model = SVC(kernel="linear", degree=7)
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print(acc)
    return df

def bigger_training_set_svm():
    test_df = matches_wobo_2019_2020_svm()
    train_df_1 = matches_wobo_2018_2019_svm()
    train_df_2 = matches_wobo_2017_2018_svm()
    train_frames = [train_df_1, train_df_2]
    train_df = pd.concat(train_frames)
    print(len(test_df))
bigger_training_set_svm()


def matches_bo_svm():
    path = "C:/Users/Jonathan/PycharmProjects/Machine-Learning---Soccer-Stats/Data/matches-bo-2019-2020.csv"
    df = pd.read_csv(path, sep=',', dtype=str)
    df.drop(['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'Referee'], axis=1, inplace=True)

    # y is FTR, Full Time Result. Denoted with a H/D/A   for Home win/ Draw/ Away win
    # Essentially, This will discover if teams win more often when they play as home or away.
    # can try a  differeent y later. Maybe creating separate testing data with only team matches.
    # can add referee id later to include referees
    # change the values for HTR, Half time results.
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'H'] = 1.0
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'D'] = 0.0
    df.loc[:, 'HTR'][df.loc[:, 'HTR'] == 'A'] = -1.0
    X = np.array(df.drop(['FTR'], axis=1))
    Y = np.array(df.loc[:, ['FTR']])
    # Change string values to numbers for y. H/D/A
    ytemp = []
    for r in Y:
        if (r == 'H'):
            ytemp.append(1)
        elif (r == 'D'):
            ytemp.append(0)
        elif (r == 'A'):
            ytemp.append(-1)
    Y = ytemp

    # Normilize data
    datax = X
    datax = datax.astype(np.float)
    s = MinMaxScaler()
    s.fit(datax)
    datax = s.transform(datax)

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=1)

    model = SVC(kernel="rbf", degree=1)
    model.fit(train_x, train_y)

    pred_y = model.predict(test_x)

    acc = accuracy_score(test_y, pred_y)
    print(acc)

    #Plot
    data_x = df  #used for the plot only
    data_y = df.loc[:,['FTHG'] ]
    ytemp = []
    for i in range(data_y.index.stop):
        ytemp.append(i + 1)
    #created a count for each of the games
    data_y.loc[:, 'FTHG'][data_y.loc[:, 'FTHG'] != False] = ytemp
    #MaxH, MaxD, MaxA, AvgH, AvgD, AvgA
    odds_data = df.loc[:,["MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA"]]
    plt.scatter(data_y.iloc[:,0], odds_data.iloc[:, 5], c = 'coral', s=50, cmap='autumn')
    plt.title('Match Results')
    plt.xlabel('Single Match')
    plt.ylabel('Odds of Home win')
    plt.show()

def past_matches_svm():
    path = "C:/Users/Jonathan/PycharmProjects/Machine-Learning---Soccer-Stats/Code/models/2018-2019.csv"
    df = pd.read_csv(path, sep=',', dtype=str)
    df.drop(['HTR', 'Date', 'HomeTeam', 'AwayTeam', 'Referee'], axis=1, inplace=True)
    X = np.array(df.drop(['FTR'], axis=1))
    Y = np.array(df.loc[:, ['FTR']])
    # Change string values to numbers for y. H/D/A
    ytemp = []
    for r in Y:
        if (r == 'H'):
            ytemp.append(1)
        elif (r == 'D'):
            ytemp.append(0)
        elif (r == 'A'):
            ytemp.append(-1)
    Y = ytemp

    # Normilize data
    datax = X
    datax = datax.astype(np.float)
    s = MinMaxScaler()
    s.fit(datax)
    datax = s.transform(datax)

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=1)

    model = SVC(kernel="rbf", degree=1)
    model.fit(train_x, train_y)

    pred_y = model.predict(test_x)

    acc = accuracy_score(test_y, pred_y)
    print(acc)

