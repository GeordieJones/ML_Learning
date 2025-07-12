import numpy as np

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
        rgen = np.random.RamdomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale= 0.01, size=x.shape[1])
        self.b_ = np.float_(0.)
        self.errors = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x,y):
                update = self.eta*(target-self.predict(xi))
                self.w_ = update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, x):
        return np.dot(x, self.w_) + self.b_
    
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0,1,0)