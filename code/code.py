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

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

label=le.fit_transform(data["Sex"])
print(label)

data["Sex"]=label

print(data)

#OneHotEncoder

from sklearn.preprocessing import OneHotEncoder

encode=OneHotEncoder()

Features=encode.fit_transform(data[["Embarked"]]).toarray()

feature=pd.DataFrame(Features,columns=["C","Q","S"])

pd.concat([data,feature],axis=1)

print(data)


import matplotlib.pyplot as plt
def bargraph():
    plt.bar(data["Sex"],data["Age"])
    plt.xlabel("sex")
    plt.ylabel("Age")
    plt.title("Age vs Sex")
    return plt.show()
bargraph()

#Violinplot
import seaborn as sns
def violnplt():
    ax = sns.violinplot(x="Pclass", y="Age", hue="Sex",data=data,kde=True)
    plt.title("Age Vs Pclass")
    return plt.show()
violnplt()

