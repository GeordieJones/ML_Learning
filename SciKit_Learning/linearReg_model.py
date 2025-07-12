import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perception:
    '''
    parameters:
    eta: learning rate (1-0)
    n_iter: interates over dataset
    random_state: random weight initialization
    w[1D]: weights after filtering
    b_: biases after filtering
    '''
    def __init__(self, eta = 0.01, n_iter = 50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        '''
        fit training set
        x = [n_examples, n_features]
        y = [n_examples]
        '''
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale= 0.01, size=x.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x,y):
                update = self.eta*(target-self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, x):
        return np.dot(x, self.w_) + self.b_
    
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0,1,0)



s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(s, header=None, encoding='utf-8')

y = df.iloc[0:100,4].values
y = np.where(y== 'Iris-setosa', 0,1)

x = df.iloc[0:100, [0,2]].values

'''plt.scatter(x[:50,0], x[:50, 1],color = 'red', marker='o', label = 'Setosa')
plt.scatter(x[50:100,0], x[50:100, 1],color = 'blue', marker='s', label = 'Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()'''

ppn = Perception(eta=0.1, n_iter=10)
ppn.fit(x,y)
'''plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('# of updates')
plt.show()'''


def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers = ('o','s','^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() -1, X[:,0].max() +1
    x2_min, x2_max = X[:,1].min() -1, X[:,1].max() +1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max, resolution),
                        np.arange(x2_min,x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],
                    y=X[y==cl,1],
                    alpha = 0.8,
                    c=colors[idx],
                    marker = markers[idx],
                    label = f'Class {cl}',
                    edgecolor = 'black')

plot_decision_regions(x,y, classifier=ppn)
plt.xlabel('Sepel length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()