import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
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
  "SVM":SVC(kernel="rbf",random_state=42),
  "randomforest":RandomForestClassifier(random_state=42,n_estimators=10),
  "logisticRegression":LogisticRegression(random_state=42)
}

model["DTree"].fit(X_train,y_train)
model["SVM"].fit(X_train,y_train)
model["randomforest"].fit(X_train,y_train)
model["logisticRegression"].fit(X_train,y_train)
print(accuracy_score(y_test,model["DTree"].predict(X_test)))
print(accuracy_score(y_test,model["SVM"].predict(X_test)))
print(accuracy_score(y_test,model["randomforest"].predict(X_test)))
print(accuracy_score(y_test,model["logisticRegression"].predict(X_test)))
