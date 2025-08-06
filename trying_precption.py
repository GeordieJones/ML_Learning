import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perception:

    def __init__(self, eta = 0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float(0.)
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*( target- self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self


    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_


    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1,0)
    
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(s, header=None, encoding='utf-8')
y = df.iloc[0:100,4].values
y = np.where(y== 'Iris-setosa', 0,1)

x = df.iloc[0:100, [0,2]].values

ppn = Perception(eta=0.01, n_iter=10)
ppn.fit(x,y)

plt.plot(range(1,len(ppn.errors)+1), ppn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('# of updates')
plt.show()



'''plt.scatter(x[:50,0], x[:50, 1],color = 'red', marker='o', label = 'Setosa')
plt.scatter(x[50:100,0], x[50:100, 1],color = 'blue', marker='s', label = 'Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()'''