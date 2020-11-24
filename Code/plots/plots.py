import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def plot_possession():
    path = "C:/Users/Jonathan/PycharmProjects/Machine-Learning---Soccer-Stats/Data/Combined_data.csv"
    df = pd.read_csv(path, sep=',', dtype=str)
    temp_df = df
    df = df.drop(["Squad"], axis=1)
    data = np.array(df)
    data = data.astype(np.float)
    scaled_df = pd.DataFrame(data, columns=df.columns)
    partial_df = scaled_df.drop(["ATT", "MID", "DEF", "GA", "GF", "GDiff","MP","Pts","OVR"], axis=1)
    partial_df['Squad'] = temp_df.loc[:,"Squad"]
    print(scaled_df.head())
    ax = sns.pairplot(partial_df, kind='reg')
    plt.show()

plot_possession()