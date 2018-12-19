import math
import numpy as np

class LogisticRegression:
    def __init__(self, alpha, l, n):
        self.thetas = np.zeros((n, 1), dtype= float)
        self.m = 0
        self.n = n
        self.cost = np.zeros((n, 1), dtype= float)
        self.alpha = alpha
        self.l = l

    def sigmoid(self, f):
        return 1 / (1 + math.exp(-f))

    def hypothesis(self, x):
        res = self.thetas.T.dot(x.T)[0]
        for i, ind in enumerate(res):
            res[i] = self.sigmoid(ind)
        return res

    def costs(self, x, y):
        self.cost = ((self.hypothesis(x)[np.newaxis].T - y).T.dot(x)).T + (self.l / self.m) * self.thetas

    def GD(self):
        self.thetas -= self.alpha * (1/self.m) * self.cost

    def calculate(self, x, y):
        self.costs(x, y)
        self.GD()
        print("costs:", end=" ")
        for cos in self.cost:
            print(cos[0], ",", end=" ")
        print(" ")
        self.cost = np.zeros((self.n, 1), dtype= float)
        print("thetas:", end=" ")
        for theta in self.thetas:
            print(theta[0], ",", end=" ")
        print('\n')

    def run(self, i, x, y):
        self.m = len(y)
        for _ in range(i):
            self.calculate(x, y)

    def predict(self, x):
        return self.hypothesis(x)
