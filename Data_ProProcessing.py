import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from Neural_Network import NeuralNetwork

df = pd.read_csv('data/diabetes.csv')
df.replace(0, df.median(), inplace = True)
X = df.drop(['Outcome'], axis=1).values
y = df['Outcome'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.eye(2)[y]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alpha = 0.1

model = NeuralNetwork([8, 12, 2], alpha = alpha)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("\nTest Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))