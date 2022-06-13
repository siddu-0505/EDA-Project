import pandas as pd
data=pd.read_csv("C:\\Users\\User\\Documents\\GitHub\\EDA-Project\\data\\Titanic-Train-Data.csv")
print(data)

print(data.columns)

print(data.describe())

print(data.isnull().sum())

data.dropna(inplace=True)

data.corr()

print(data)

pd.get_dummies(data["Sex"])

print(data)

print(data.head())