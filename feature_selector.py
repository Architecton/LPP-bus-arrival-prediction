from sklearn.preprocessing import PolynomialFeatures
from regression_analysis_lib import LinearLearner
import numpy as np
import scipy
import sklearn.metrics
import random
import os
import time
import inspyred

## FEATURE SELECTION WITH A GENETIC ALGORITHM ##


# Simple test
if __name__ == '__main__':
    print('Selecting best features for each bus line with a genetic algorithm. This may take a while.')
    # Go over data folders for bus lines.
    for line_dir in os.listdir('./data_processing/bus-line-data-train'):
        X = np.load('./data_processing/bus-line-data-train/' + line_dir + '/predictors_train.npy')
        y = np.load('./data_processing/bus-line-data-train/' + line_dir + '/target_train.npy')

        # Take one-fifth of data as test data.
        SIZE_TEST = X.shape[0] // 5
        # Get test data and training data indices.
        test_idx = np.array(random.sample(range(X.shape[0]), SIZE_TEST))
        train_idx = np.array(list(set(range(X.shape[0])) - set(test_idx)))
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]


        # generate_sel: function for generating random chromosomes.
        def generate_sel(random, args):
            bits = args.get('num_bits', 8)
            return [random.choice([True, False]) for i in range(bits)]


        # fintess function.
        @inspyred.ec.evaluators.evaluator
        def fitness(sel, args):
            poly = PolynomialFeatures(degree=1)
            # Get sparse matrix representation.
            new_X_train = poly.fit_transform(X_train[:, sel])
            new_X_test = poly.fit_transform(X_test[:, sel])
            Xsp = scipy.sparse.csr_matrix(new_X_train)
            # Define a LinearLearner instance.
            lr = LinearLearner(lambda_=1.5)
            # Get prediction function.
            linear = lr(Xsp, y[train_idx])
            # Get results.
            res = np.empty(len(y[test_idx]), dtype=float)
            for res_i, row in enumerate(new_X_test):
                res[res_i] = linear(row)

            # Return MAE.
            return -sklearn.metrics.mean_absolute_error(y[test_idx], res)


        rand = random.Random()
        rand.seed(int(time.time()))
        ga = inspyred.ec.GA(rand)
        ga.observer = inspyred.ec.observers.stats_observer
        ga.terminator = inspyred.ec.terminators.evaluation_termination

        # Get final population. Use parallelized evaluation.
        final_pop = ga.evolve(evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
                              mp_evaluator=fitness,
                              generator=generate_sel,
                              max_evaluations=10,
                              num_elites=1,
                              pop_size=10,
                              mp_num_cpus=8,
                              num_bits=X.shape[1])

        # Sort individuals by fitness value in descending order.
        final_pop.sort(reverse=True)

        # Get chromosome of individual with highest fitness and save as a feature
        # selection vector to be used with linear regression.
        res = final_pop[0].candidate
        np.save('./data_processing/bus-line-data-train/' + line_dir + '/feature_sel.npy', res)

    print('Done.')
