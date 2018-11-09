import numpy
from scipy.optimize import fmin_l_bfgs_b
import scipy.sparse
import scipy.sparse as sp
import numpy as np


# append_ones: append ones to the right side of the feature matrix.
def append_ones(X):
    # If matrix is sparse...
    if sp.issparse(X):
        return sp.hstack((np.ones((X.shape[0], 1)), X)).tocsr()
    # Else...
    else:
        return np.hstack((np.ones((X.shape[0], 1)), X))


# hl:
def hl(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    return x.dot(theta)


# cost_grad_linear:
def cost_grad_linear(theta, X, y, lambda_):
    # do not regularize the first element
    sx = hl(X, theta)
    j = 0.5*numpy.mean((sx-y)*(sx-y)) + 1/2.*lambda_*theta[1:].dot(theta[1:])/y.shape[0]
    grad = X.T.dot(sx-y)/y.shape[0] + numpy.hstack([[0.],lambda_*theta[1:]])/y.shape[0]
    return j, grad


# linearLearner:
class LinearLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        # Append ones to the data matrix.
        X = append_ones(X)

        th = fmin_l_bfgs_b(cost_grad_linear,
            x0=numpy.zeros(X.shape[1]),
            args=(X, y, self.lambda_))[0]

        return LinearRegClassifier(th)


# LinearRegClassifier
class LinearRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = numpy.hstack(([1.], x))
        return hl(x, self.th)

# test with basic dataset.
if __name__ == "__main__":

    verbose = 0

    import Orange

    # iris: get feature matrix and target variable values of the iris data set.
    def iris():
        # Get data from Orange
        data = Orange.data.Table("iris")
        # Return the features matrix and the target variable results vector.
        return data.X, data.Y

    # Get predictor and target matrices.
    X, y = iris()

    # Get sparse matrix representation.
    Xsp = scipy.sparse.csr_matrix(X)

    # Define a LinearLearner instance.
    lr = LinearLearner(lambda_=1.)

    # Get prediction function.
    linear = lr(Xsp,y)

    for a in X:
        print(linear(a))