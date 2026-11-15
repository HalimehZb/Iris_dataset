import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

dataset=pd.read_csv("iris.data.csv")

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
g=LabelEncoder()
new_y=g.fit_transform(Y)

z=StandardScaler()
new_x=z.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(new_x, new_y, test_size=0.30, random_state=42 ,shuffle=True)


model={
  "DTree":DecisionTreeClassifier(random_state=42),
  "SVM":SVC(kernel="rbf",random_state=42)
}