#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats.stats import pearsonr
import pandas as pd
from learning.helpers import cost, AU, MU
import timeit


class NMF:
    """Non-Negative Matrix Factorization to predict grades.

    Parameters:
    -----------
    W : preloaded W matrix
    H : preloaded H matrix

    Usage:
    -----
    Can be used with preloaded data:

        from data import unicamp
        from nmf import NMF
        u = unicamp.load()
        model = NMF(u.W, u.H)
        test = [10, 5, 0, 0, 9, 8, 0, 0, 0, 0, 8, 0]
        model.predict(test)

    Can also be used with new data(might take a while to fit):

        from data import unicamp
        from nmf import NMF
        u = unicamp.load()
        model = NMF()
        model.fit(u.data, 8)
        test = [10, 5, 0, 0, 9, 8, 0, 0, 0, 0, 8, 0]
        model.predict(test)

    the generated W and H can be accessed with:
    model.W
    model.H

    To extract the whole predicted matrix:
    model.get_V()
    """

    def __init__(self, W=[], H=[], verbose=0):
        # W and H can be preloaded if you just want to predict grades using
        # a known dataset
        self.verbose = verbose
        self.W = np.array(W)
        self.H = np.array(H)
        # if a H was loaded, extract R from it
        if self.H.any():
            self.R = len(self.H)
        # else, use R=1
        else:
            self.R = 1

    def fit(self, X, R=1, steps=5000, alpha=0.004, beta=0.002, gamma=0.0002, eC=1, alg=0):
        """Fit method.

        Parameters:
        ----------
        X: dataset(matrix) to be factorized
        R: number of features to find
        steps: maximum iterations to be perfermed when trying to minimize the distance between X and W.H
        alpha: rate of approaching the minimum distance
        beta: regularizing variable for W
        gamma: regularizing variable for H
        eC: minimum error required
        alg: which algorithm to start with
        """
        # sets instance vars
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eC = eC
        self.steps = steps
        np.random.seed(0)
        if not self.W.any():
            self.W = np.random.rand(len(X), R)
        if not self.H.any():
            self.H = np.random.rand(R, len(X[0]))

        # optimization variable to alternate between H and W
        alternate = 1

        # Switch threshold
        # used to decide when to switch algorithms
        switch_threshold = (self.eC ** 2) * X.size / 200

        # benchmark
        start = timeit.default_timer()

        # iterates 'steps' times or until cost < eC
        for step in range(self.steps):

            # calculates the cost only once per 2 iterations
            if alternate == 1:
                e = cost(X, self.W, self.H, self.beta, self.gamma, self.R)
            if step == 0:
                # Manually set the first prev_e so that it is always bigger than e
                prev_e = e + 100

            # if the error is less than the stopping point, break
            self.error_fit = e
            if e < eC:
                break

            # At every 100 iterations
            if step % 100 == 0:

                # If the current algorithm starts to get stuck, change to the next or just end the calculation
                if prev_e - e < switch_threshold:
                    alg += 1
                prev_e = e

                # verbose benchmarking
                if self.verbose != 0:
                    stop = timeit.default_timer()
                    print("Computation time: " + str(stop - start))
                    start = timeit.default_timer()
                    print("Current error: " + str(e))
                    print("Current iter: " + str(step))

            # Alternate betwen evaluations
            if alternate == 1:
                alternate = 0
            else:
                alternate = 1

            # Start using MU
            if alg == 0:
                self.W, self.H = MU(X, self.W, self.H, alternate, self.R)

            # If MU gets stuck, change to AU
            elif alg == 1:
                self.W, self.H = AU(X, self.W, self.H, self.alpha, self.beta, self.gamma, alternate, self.R)

            # If AU gets stuck, break
            else:
                break
        return

    def predict(self, X, steps=50000, alpha=0.004, beta=0.002, gamma=0.0002, eC=0.02):
        """Predict method.

        Parameters:
        ----------
        X: 2D of users to be predicted (can also handle 1D arrays)
        steps: maximum iterations to be perfermed when trying to minimize the distance between X and W.H
        alpha: rate of approaching the minimum distance
        beta: regularizing variable for W
        gamma: regularizing variable for H
        eC: minimum error required

        Returns:
        --------
        numpy 2D array with predicted users' grades
        """

        # Accept both single and multiple predictions at the same time
        try:
            X[0][0]
            X = np.array(X)
        except:
            X = np.array([X])

        # Initialize W and use the loaded H
        np.random.seed(0)
        W = np.random.rand(len(X), self.R)

        # Keep track of previous error
        prev_e = 0
        switch_threshold = (eC ** 2) * X.size / 200

        # Do the same thing as in fit(), but only approximate for W
        for step in range(steps):
            W, H = AU(X, W, self.H, alpha, beta, gamma, 1, self.R)
            e = cost(X, W, self.H, beta, gamma, self.R)
            self.error_predict = e
            if e < eC or abs(prev_e - e) < switch_threshold:
                break

        # return the predicted matrix
        return np.dot(W, self.H)

    def get_V(self):
        # Get the whole fitted matrix (V = WH ≃ X)
        return np.dot(self.W, self.H)


class NB_CF():
    """Neighborhood-Based Collaborative Filtering.

    This is pretty much useless with my dataset
    but was easy to make, so why not

    Usage:
        from Data import Unicamp
        from Fun.ML import NB_CF
        u = Unicamp.load()
        model = NB_CF()
        active = [10, 5, 0, 0, 9, 8, 0, 0, 0, 0, 8, 0]
        model.fit(active, u.dataset)
        model.predict(0)
    """

    def __init__(self):
        pass

    def fit(self, active, dataset, n_neighbors=20):
        """Fit method.

        Parameters:
        ----------
        active: user to be predicted
        dataset: dataset with all users
        n_neighbors: number of closest neighbors to use in prediction
        """
        self.dataset = dataset
        self.active = active

        # calculate the Pearson correlation between every database user and active
        self.calc_correlation()

        # dirty way to append the correlation to the database.
        frame = pd.DataFrame(dataset)
        correlation = pd.DataFrame(self.correlation)
        frame['correlation'] = correlation
        frame = frame.sort_values(by=['correlation'], ascending=False)[:n_neighbors]
        self.neighbors = frame.as_matrix()
        return

    def predict(self, item):
        """Predict method.

        Parameters:
        ----------
        item : index of the item to be predicted for the fitted user

        Returns:
        --------
        prediction (float)
        """

        def n_mean(user):
            # returns the mean of the user's grades(only the ones > 0)
            return np.array([x for x in user[:-1] if x > 0]).mean()

        def sum_W(neighbors):
            # sum of the neighbors' correlations
            return np.array([x[-1] for x in neighbors]).sum()

        def weighted_avg(neighbor, item):
            # weighted average grade of a given neighbor
            # w_avg = ((neighbor's grade on item) - (neighbor's mean)) * (neighbor's correlation)
            if neighbor[item] == 0:
                return 0
            return (neighbor[item] - n_mean(neighbor)) * neighbor[-1]

        def prediction(item):
            # calculate the weighted average of every neighbor, sum it all
            # divide by the sum of the correlations and then add the active user's mean

            pred = 0
            for neighbor in self.neighbors:
                pred = pred + weighted_avg(neighbor, item)

            pred = n_mean(self.active) + pred / sum_W(self.neighbors)

            return pred

        # Return the prediction for the item
        return prediction(item)

    def calc_correlation(self):
        # Uses scipy's pearson correlation. Might create my own to achieve better results

        # indexes of the active's graded items
        items_active = [counter for counter, item in enumerate(self.active) if item > 0]

        corr = []
        for user in self.dataset:
            # indexes of the user's graded items
            items_user = [counter for counter, item in enumerate(user) if item > 0]
            # indexes that both active and user have in common
            items_both = [counter for counter in items_user if counter in items_active]
            # actual items that active and user have in common
            active_items = [item for counter, item in enumerate(self.active) if counter in items_both]
            user_items = [item for counter, item in enumerate(user) if counter in items_both]
            # calculate the correlation
            new_corr = pearsonr(active_items, user_items)
            # if there are too few items in commom, or if the correlation is NaN, set it as 0
            if len(active_items) <= 1 or np.isnan(new_corr[0]):
                new_corr = 0, 1

            # append the correlation to the list
            corr.append(new_corr[0])
        self.correlation = corr
