from .nmf import NMF
from data import unicamp
import timeit
import random


def full_test(percentage, data=unicamp.load().data):
    # Function used to as cross verification
    # The end user does not interact with it
    features = []

    # Create test set
    perc = percentage / 100
    n_train = int(len(data) * (1 - perc))
    test = data[n_train:]
    print("Creating test set.")
    test = create_test(test)

    # Fit and test all features
    for n in range(2,11):
        perc, error = run(percentage, n, data, test)
        features.append([n, perc, error])

    print()
    print()
    print(features)
    return features


def run(percentage, R=12, uni=unicamp.load().data, test=[]):
    # Function that runs and evaluates the results
    # Generate training set
    percentage = percentage / 100
    n_train = int(len(uni) * (1 - percentage))
    n_test = len(uni) - n_train
    train = uni[: n_train]
    print()
    # Train the model
    print("Training with " + str(n_train) + " elements, " + str(R) + " features.")
    nmf = NMF()
    start = timeit.default_timer()
    nmf.fit(train, R, alg=1)
    stop = timeit.default_timer()
    print()
    # Benchmark stuff
    print("Took " + str(stop - start) + " seconds.")
    print("Fit error: " + str(nmf.error_fit))
    print()
    errors = 0
    # Test every item in the test set and Benchmark its results
    print("Starting test.")
    for item, deleted, index in test:
        start = timeit.default_timer()
        pred = nmf.predict(item)[0]
        stop = timeit.default_timer()
        print("Took " + str(stop - start) + " seconds to predict.")
        prediction = int(round(pred[index]))
        if prediction == deleted:
            print("Right.")
        else:
            print("Wrong, got " + str(prediction) + " in " + str(deleted))
            errors = errors + 1

    # Return the results
    print()
    print("Total errors: " + str(errors) + " out of " + str(n_test))
    print("Percentage: " + str((n_test - errors) / n_test))

    return (n_test - errors) / n_test, nmf.error_fit


def create_test(data):
    # Function that creates the test set
    test = []
    for item in data:
        deleted = 0
        index = 0
        while True:
            # Select a nonzero value, delete it and save the original value
            to_delete = random.choice([0, 1, 2, 4, 5, 8, 10, 11])
            if item[to_delete] != 0:
                deleted = item[to_delete]
                index = to_delete
                item[to_delete] = 0
                break
        test.append([item, deleted, index])

    # return the test set
    return test
