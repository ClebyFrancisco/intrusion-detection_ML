import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("../data/1_ddos/1_ddos_1perc.csv", header=0, na_values=["?"]).dropna()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc}")
print(conf_mat)