import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

dataset = pd.read_csv("../data/1_ddos/1_ddos_10perc.csv", header=0, na_values=["?"]).dropna()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

conf_matr = confusion_matrix(y_test, y_pred)
print("conf_matr:")
print(conf_matr)

print("************ Desempenho KNN - 1_ddos_10perc **********")
print("Acuracia:", accuracy_score(y_test, y_pred)*100)
print("Precision:", precision_score(y_test, y_pred)*100)
print("Recall:", recall_score(y_test, y_pred)*100)
print("F1_Score:", f1_score(y_test, y_pred)*100)
print("AUC:", roc_auc_score(y_test, y_pred)*100)