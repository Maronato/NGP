from .nmf import NMF
from data import unicamp
import timeit
import random


def full_test(percentage, data=unicamp.load().data):

    features = []

    perc = percentage / 100
    n_train = int(len(data) * (1 - perc))
    test = data[n_train:]
    print("Creating test set.")
    test = create_test(test)

    for n in range(1, 16):
        perc, error = run(percentage, n, data, test)
        features.append([n, perc, error])

    print()
    print()
    print(features)
    return features


def run(percentage, R=12, uni=unicamp.load().data, test=[]):
    percentage = percentage / 100
    n_train = int(len(uni) * (1 - percentage))
    n_test = len(uni) - n_train
    train = uni[: n_train]
    print()
    print("Training with " + str(n_train) + " elements, " + str(R) + " features.")
    nmf = NMF()
    start = timeit.default_timer()
    nmf.fit(train, R)
    stop = timeit.default_timer()
    print()
    print("Took " + str(stop - start) + " seconds.")
    print("Fit error: " + str(nmf.error_fit))
    print()
    errors = 0
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
    print()
    print("Total errors: " + str(errors) + " out of " + str(n_test))
    print("Percentage: " + str((n_test - errors) / n_test))

    return (n_test - errors) / n_test, nmf.error_fit


def create_test(data):
    test = []
    for item in data:
        deleted = 0
        index = 0
        while True:
            to_delete = random.randint(0, len(item) - 1)
            if item[to_delete] != 0:
                deleted = item[to_delete]
                index = to_delete
                item[to_delete] = 0
                break
        test.append([item, deleted, index])

    return test
