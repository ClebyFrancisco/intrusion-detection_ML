import pandas as pd

import numpy as np
from numpy import array

from sklearn import model_selection
from sklearn.model_selection import cross_val_predict,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import *

ds = array(pd.read_csv("../data/1_ddos/1_ddos_1perc.csv"))


# x -> features (coluna 0-77)
x = ds[:, 0:77]
# y -> labels (coluna 78)
y = ds[:, 78]


clf = SVC(kernel='rbf', C=10, gamma=1)
clf_pred = cross_val_predict(clf, x, y, cv=10)
conf_clf = confusion_matrix(y, clf_pred)

print("************ Desempenho SVM **********")
print("Acuracia:", accuracy_score(y, clf_pred)*100)
print("AUC:", roc_auc_score(y, clf_pred)*100)
print("Precision:", precision_score(y, clf_pred)*100)
print("Recall:", recall_score(y, clf_pred)*100)
print("F1_Score:", f1_score(y, clf_pred)*100)