import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize


class Adaboost(object):
    def __init__(self):
        self.trees = None
        self.trees_weight = None
        self.loss = None
        self.weights = None

    def fit(self, X, y, n_iter=10):

        X = np.float64(X)
        N = len(X)

        self.trees = np.zeros(shape=(n_iter, ), dtype=object)
        self.trees_weight = np.zeros(shape=(n_iter, ))
        self.loss = np.zeros(shape=(n_iter, ))
        self.weights = np.zeros(shape=(n_iter, N))

        self.weights[0] = np.ones(N) / N

        for i in range(n_iter):
            weight = self.weights[i]  # Init weight
            tree = DecisionTreeRegressor(max_depth=1,
                                         max_leaf_nodes=2)  #Weak learner
            tree = tree.fit(X, y, sample_weight=weight)
            tree_pred = tree.predict(X)
            loss = np.abs(y - tree_pred)  # Loss value
            den = np.max(loss)
            avg_loss = np.sum(np.dot(loss / den,
                                     weight))  # Loss function linear form
            tree_weight = avg_loss / (1 - avg_loss)
            new_weight = (weight * (tree_weight**(1 - avg_loss))
                          )  # Update the weight matrix

            if i + 1 < n_iter:
                self.weights[i + 1] = new_weight / np.sum(new_weight)
            #  Update the values in my object
            self.trees[i] = tree
            self.trees_weight[i] = tree_weight
            self.loss[i] = avg_loss

    def pred(self, X):
        prediction = []
        for obs in X:

            trees_pred = [
                tree.predict(obs.reshape(1, -1)) for tree in self.trees
            ]  # need to reshape because one sample, and use the tree saved to predict

            def func(x):  # create function you need to minimize
                res = np.sum(
                    np.array([
                        1 - self.trees_weight[i] if trees_pred[i] < x else 0
                        for i in range(len(trees_pred))
                    ]))
                res -= 0.5 * np.sum(np.log(1 - self.trees_weight))
                return res

            starting_guess = np.min(
                trees_pred)  #Initialize at min or the predictions
            prediction.append(minimize(func, [starting_guess]).x[0])
        return prediction


if __name__ == "__main__":
    boston = load_boston()
    X, y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = Adaboost()
    model.fit(X_train, y_train, n_iter=100)
    y_pred = model.pred(X_test)
    print(mean_absolute_error(y_test, y_pred))