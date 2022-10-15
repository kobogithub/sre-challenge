import pickle
import time
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

x_train = pd.read_csv('datasets/x_train.csv', header = None).to_numpy()
y_train = np.ravel(pd.read_csv('datasets/y_train.csv', header = None).to_numpy())
x_test = pd.read_csv('datasets/x_test.csv', header = None).to_numpy()
y_test = np.ravel(pd.read_csv('datasets/y_test.csv', header = None).to_numpy())

print(y_test)

model = pickle.load(open('pickle_model.pkl', 'rb'))
start = time.time()
y_pred = model.predict(x_test)
end = time.time()
print("Tiempo en predicción:", end - start, "[s]")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))