import sklearn as sk
import pandas as pd

#Analyze Combine_data.csv first.
#classes will be [w, d, l] = [0,0,0]
data = pd.read_csv('../../Data/Combined_data.csv', sep=',', dtype=str)
print(data)