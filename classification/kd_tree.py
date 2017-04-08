from sklearn.neighbors import KDTree
import numpy as np

class classifier:

    def __init__(this, X, Y):
        this.X = X
        this.Y = Y
        this.tree = KDTree(X)

if __name__ == '__main__':
    c = classifier(np.random.random((10, 2)), np.random.randint(0, 1))
