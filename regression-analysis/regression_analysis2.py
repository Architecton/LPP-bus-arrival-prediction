from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model
import numpy as np


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