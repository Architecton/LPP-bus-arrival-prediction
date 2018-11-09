import scipy.sparse as sp
import numpy
from scipy.optimize import fmin_l_bfgs_b
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


## Function for regression modelling using the scikit-learn library ##

# regression_analysis: perform regression analysis on feature matrix X and target matrix y.
# Construct regression matrix of degree specified by the polynomial_degree argument. Accepts optional
# alpha parameter for the regression function.
def regression_analysis(x, y, polynomial_degree, model_func, param):
    poly = PolynomialFeatures(degree=polynomial_degree)
    new_x = poly.fit_transform(x)

    # If regression function takes a parameter
    if param:
        reg = model_func(alphas=np.linspace(0.1, 10, 20), cv=3)
    else:
        reg = model_func()

    # Fit polynomial
    reg.fit(new_x, y)
    # Return regression matrix and regression coefficients
    return new_x, reg.coef_

######################################################################

#  Regression model written by assistants at FRI

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