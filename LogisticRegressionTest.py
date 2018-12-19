import LogisticRegression as lor
import numpy as np
import pandas as pd

x3 = np.array([[1, 1, 1], [1, 1.5, 2], [1, 6, 6], [1, 7, 6]], dtype=float)
y3 = np.array([[1], [1], [0], [0]], dtype=float)

x2 = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 8]], dtype=float)
y2 = np.array([[0], [0], [0], [1], [1], [1]], dtype=float)

LOR = lor.LogisticRegression(18, 0, 3)
LOR.run(100000, x3, y3)
print(LOR.predict(np.array([[1, 1, 1]])))

LOR2D = lor.LogisticRegression(100, 0, 2)
LOR2D.run(100000, x2, y2)
print(LOR2D.predict(np.array([[1, 4]])))

#This takes a very long time...at least on my mediocre laptop
pd_train = pd.read_csv('train.csv')

pd_train.drop(["Name", "Ticket", "Fare", "Cabin"], 1, inplace=True)
avg_age = sum(pd_train['Age'].dropna(0)) / len(pd_train['Age'].dropna(0))
pd_train.fillna(avg_age, inplace=True)
pd_train.replace(['male', 'female', 'C', 'S', 'Q'], [0, 1, 1, 2, 3], inplace=True)

y = np.array(pd_train['Survived'])[np.newaxis].T

pd_train.drop(['Survived'], 1, inplace=True)
x = np.array(pd_train)
ones = np.ones((891, 1), dtype=float)
x = np.hstack((ones, x))

pd_test = pd.read_csv('test.csv')

pd_test.drop(["Name", "Ticket", "Fare", "Cabin"], 1, inplace=True)
avg_age = sum(pd_test['Age'].dropna(0)) / len(pd_test['Age'].dropna(0))
pd_test.fillna(avg_age, inplace=True)
pd_test.replace(['male', 'female', 'C', 'S', 'Q'], [0, 1, 1, 2, 3], inplace=True)

x_test = np.array(pd_test)
ones_test = np.ones((418, 1), dtype=float)
x_test = np.hstack((ones_test, x_test))

LOR_titanic = lor.LogisticRegression(0.00003, 5000, 8)
LOR_titanic.run(1000000, x, y)
print(LOR_titanic.predict(x_test))