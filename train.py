from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from dt import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

dt = DecisionTree()

dt.fit(X_train, y_train)
predictions = dt.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)

# final accuracy of 0.9210526315789473
print(acc)