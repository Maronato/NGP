import numpy
from scipy.stats.stats import pearsonr
import pandas as pd


class NMF:
    '''
        non-negative matrix factorization to predict grades

        Can be used with preloaded data:

            from Data import Unicamp
            from Fun.ML import NMF
            u = Unicamp.load()
            model = NMF(u.W, u.H)
            test = [10, 5, 0, 0, 9, 8, 0, 0, 0, 0, 8, 0]
            model.predict(test)

        Can also be used with new data(might take a while to fit):

            from Data import Unicamp
            u = Unicamp.load()
            model = NMF()
            model.fit(u.dataset, 8)
            test = [10, 5, 0, 0, 9, 8, 0, 0, 0, 0, 8, 0]
            model.predict(test)

        the generated W and H can be accessed with:
        model.W
        model.H

        To extract the whole predicted matrix:
        model.get_V()
    '''

    def __init__(self, W=[], H=[]):
        # W and H can be preloaded if you just want to predict grades using
        # a known dataset
        self.W = numpy.array(W)
        self.H = numpy.array(H)
        # if a H was loaded, extract R from it
        if self.H.any():
            self.R = len(self.H)
        # else, use R=1
        else:
            self.R = 1

    def cost(self, X, W, beta, gamma):
        # Cost function.
        # Calculates the total distance between W.H and X
        # is used to break the function. For now, it's a little useless since
        # the choosen reduction function never converges to the min :/
        e = 0
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] > 0:
                    e = e + pow(X[i][j] - numpy.dot(W[i, :], self.H[:, j]), 2)
                    for r in range(self.R):
                        e = e + (beta / 2) * (pow(W[i][r], 2) + (gamma / 2) * pow(self.H[r][j], 2))
        return e

    def fit(self, X, R=2, steps=5000, alpha=0.0002, beta=0.02, gamma=0.02, eC=0.001):
        '''
            X: dataset(matrix) to be factorized
            R: number of features to find
            steps: maximum iterations to be perfermed when trying to minimize the distance between X and W.H
            alpha: rate of approaching the minimum distance
            beta: regularizing variable for W
            gamma: regularizing variable for H
            eC: minimum error required
            W: feature weights, users
            H: feature weights, items
        '''
        # sets instance vars
        self.R = R
        if not self.W.any():
            self.W = numpy.random.rand(len(X), R)
        if not self.H.any():
            self.H = numpy.random.rand(R, len(X[0]))

        # iterates 'steps' times until cost < eC
        for step in range(steps):
            for i in range(len(X)):
                for j in range(len(X[i])):

                    # Only evaluate WH if the current grade is set
                    if X[i][j] > 0:

                        # Distance between X and WH
                        eij = X[i][j] - numpy.dot(self.W[i, :], self.H[:, j])

                        # Not very efficient way of minimizing the error
                        # reduce the error for every feature
                        for r in range(self.R):

                            # assuming that the error is given by e^2 = (Xij - WHij)^2
                            # We use beta and gamma to regularize the error:
                            # reg(e^2) = e^2 + (||W||^2*beta/2 + ||H||^2*gamma/2)

                            # Therefore the gradient is given, with respect to W:
                            # -2(Xij - WHij)*Hrj - beta*Wir = -2*e*Hrj - beta*Wir
                            # and, with respect to H:
                            # -2(Xij - WHij)*Wir - gamma*Hrj = -2*e*Wir - gamma*Hrj

                            # So W'ir = Wir + (alpha)*(2*e*Hrj - beta*Wir)
                            # and H'rj = Hrj + (alpha)*(2*e*Wir - gamma*Hrj)
                            # With alpha being the rate of approximation

                            self.W[i][r] = self.W[i][r] + alpha * (2 * eij * self.H[r][j] - beta * self.W[i][r])
                            self.H[r][j] = self.H[r][j] + alpha * (2 * eij * self.W[i][r] - gamma * self.H[r][j])

            # check the cost function. If min error was reached, break the loop
            if self.cost(X, self.W, beta, gamma) < eC:
                break
        return

    def predict(self, X, steps=50000, alpha=0.0002, beta=0.02, gamma=0.02, eC=0.0001):

        # Accept both single and multiple predictions at the same time
        try:
            X[0][0]
        except:
            X = [X]

        # Initialize W and use the loaded H
        W = numpy.random.rand(len(X), self.R)

        # Do the same thing as in fit(), but only approximate for W
        for step in range(steps):
            for i in range(len(X)):
                for j in range(len(X[i])):
                    if X[i][j] > 0:
                        eij = X[i][j] - numpy.dot(W[i, :], self.H[:, j])
                        for r in range(self.R):

                            # Since H is the item-features weight, there is no need to recalculate it
                            # once the model was fitted
                            W[i][r] = W[i][r] + alpha * (2 * eij * self.H[r][j] - beta * W[i][r])
            if self.cost(X, W, beta, gamma) < eC:
                break

        # return the predicted matrix
        return numpy.dot(W, self.H)

    def get_V(self):
        # Get the whole fitted matrix (V = WH ≃ X)
        return numpy.dot(self.W, self.H)


class NB_CF():

    '''
        This is pretty much useless with my dataset
        but was easy to make, so why not

        Neighborhood-Based Collaborative Filtering

        Usage:
            from Data import Unicamp
            from Fun.ML import NB_CF
            u = Unicamp.load()
            model = NB_CF()
            active = [10, 5, 0, 0, 9, 8, 0, 0, 0, 0, 8, 0]
            model.fit(active, u.dataset)
            model.predict(0)
    '''

    def __init__(self):
        pass

    def fit(self, active, dataset, n_neighbors=20):
        '''
            active: user to be predicted
            dataset: dataset with all users
            n_neighbors: number of closest neighbors to use in prediction
        '''
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
        # predict a grade from the active user
        # item should be the index of the grade on the database

        def n_mean(user):
            # returns the mean of the user's grades(only the ones > 0)
            return numpy.array([x for x in user[:-1] if x > 0]).mean()

        def sum_W(neighbors):
            # sum of the neighbors' correlations
            return numpy.array([x[-1] for x in neighbors]).sum()

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
        # Uses scipy's pearson correlation. Might create my own to achieve better
        # results

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
            if len(active_items) <= 1 or numpy.isnan(new_corr[0]):
                new_corr = 0, 1

            # append the correlation to the list
            corr.append(new_corr[0])
        self.correlation = corr
