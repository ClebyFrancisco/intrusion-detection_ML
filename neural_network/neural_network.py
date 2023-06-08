import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *

dataset = pd.read_csv("../data/1_ddos/1_ddos_10perc.csv", header=0, na_values=["?"]).dropna()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Construir o modelo de Rede Neural
model = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular as métricas de desempenho
conf_matr = confusion_matrix(y_test, y_pred)
print("conf_matr:")
print(conf_matr)

print("************ Desempenho Rede Neural - 1_ddos_10perc **********")
print("Acuracia:", accuracy_score(y_test, y_pred) * 100)
print("Precision:", precision_score(y_test, y_pred) * 100)
print("Recall:", recall_score(y_test, y_pred) * 100)
print("F1_Score:", f1_score(y_test, y_pred) * 100)
print("AUC:", roc_auc_score(y_test, y_pred) * 100)