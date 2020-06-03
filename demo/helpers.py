import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris["data"], iris["target"]
X = X + np.random.randn(*X.shape)


def precision(y, yhat, t):
    return(np.mean(y[yhat==t] == t))

def recall(y, yhat, t):
    return(np.mean(t == yhat[y==t]))

def f1(prec, rec):
    return((2*prec*rec)/(prec+rec))

def wrapper(y, yhat, t):
    precision_ = precision(y, yhat, t)
    recall_ = recall(y, yhat, t)
    f1_ = f1(precision_, recall_)
    return([precision_, recall_, f1_])

def metrics(y, yhat):
    return(np.array([wrapper(y, yhat, t) for t in sorted(np.unique(y))]))
