import scipy
import numpy as np
import regression_analysis2
import sklearn.linear_model
from sklearn.preprocessing import PolynomialFeatures
from regression_analysis import *
import results_generator
import pdb

X = np.load('data-processing/predictors_train.npy')
X_test = np.load('data-processing/predictors_test.npy')
y = np.load('data_processing/target_train.npy')
start_times_train = np.load('data-processing/start_times_train.npy')
start_times_test = np.load('data-processing/start_times_test.npy')

print('Choose regression model:')
print('1 - linear regression (scikit-learn)')
print('2 - Ridge regression (scikit-learn)')
print('3 - Lasso regression (scikit-learn)')
print('4 - Huber regression (scikit-learn)')
print('5 - Linear regression written by assistants at FRI')
while True:
    reg_model_sel = input()
    if reg_model_sel in {'1', '2', '3', '4', '5'}:
        reg_model_sel = int(reg_model_sel)
        break
    else:
        print('Invalid input. Please try again.')

if reg_model_sel == 1:
    polynomial_degree = int(input('Enter degree of polynomial to use. '))
    X_reg, coeff = regression_analysis2.regression_analysis(X, y, polynomial_degree, sklearn.linear_model.LinearRegression, False)
    res = np.sum(X_reg * coeff, axis=1)
    results_generator.print_results(start_times_train, res)
elif reg_model_sel == 2:
    polynomial_degree = int(input('Enter degree of polynomial to use. '))
    X_reg, coeff = regression_analysis2.regression_analysis(X, y, polynomial_degree, sklearn.linear_model.RidgeCV, True)
    res = np.sum(X_reg * coeff, axis=1)
    results_generator.print_results(start_times_test, res)
elif reg_model_sel == 3:
    polynomial_degree = int(input('Enter degree of polynomial to use. '))
    X_reg, coeff = regression_analysis2.regression_analysis(X, y, polynomial_degree, sklearn.linear_model.LassoCV, True)
    res = np.sum(X_reg * coeff, axis=1)
    results_generator.print_results(start_times_test, res)
elif reg_model_sel == 4:
    polynomial_degree = int(input('Enter degree of polynomial to use. '))
    poly = PolynomialFeatures(degree=polynomial_degree)
    new_X = poly.fit_transform(scipy.sparse.csr_matrix(X))
    huber = sklearn.linear_model.HuberRegressor().fit(new_X, y)
    new_X = new_X.todense()
    coeff = huber.coef_
    res = np.sum(np.matmul(new_X, coeff.reshape(len(coeff), 1)), axis=1)
    results_generator.print_results(start_times_test, res)
elif reg_model_sel == 5:
    polynomial_degree = int(input('Enter degree of polynomial to use. '))
    poly = PolynomialFeatures(degree=polynomial_degree)
    # Get sparse matrix representation.
    new_X = poly.fit_transform(X)
    Xsp = scipy.sparse.csr_matrix(new_X)
    # Define a LinearLearner instance.
    lr = LinearLearner(lambda_=1.)
    # Get prediction function.
    linear = lr(Xsp, y)

    res = np.empty(len(y), dtype=float)
    for res_i, row in enumerate(new_X):
        res[res_i] = linear(row)