import numpy as np
# usage from here:
# http://docs.cython.org/en/latest/src/tutorial/numpy.html

cimport numpy as np

DTYPE = np.double

ctypedef np.double_t DTYPE_t


def AU(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=2] H, double alpha, double beta, double gamma, int alternate, int R):
    '''
        Not quite ALS, but uses the idea of alternating between evaluations of W and H
        https://arxiv.org/pdf/1401.5226v1.pdf
        http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
    '''
    assert X.dtype == DTYPE and W.dtype == DTYPE and H.dtype == DTYPE
    cdef float eij
    for i in range(len(X)):
        for j in range(len(X[i])):

            # Only evaluate WH if the current grade is set
            if X[i][j] > 0:
                # Distance between X and WH
                eij = X[i][j] - np.dot(W[i, :], H[:, j])
                # Not very efficient way of minimizing the erimror
                # reduce the error for every feature
                for r in range(R):
                    '''
                        assuming that the error is given by e^2 = (Xij - WHij)^2
                        We use beta and gamma to regularize the error:
                        reg(e^2) = e^2 + (||W||^2*beta/2 + ||H||^2*gamma/2)

                        Therefore the gradient is given, with respect to W:
                        -2(Xij - WHij)*Hrj - beta*Wir = -2*e*Hrj - beta*Wir
                        and, with respect to H:
                        -2(Xij - WHij)*Wir - gamma*Hrj = -2*e*Wir - gamma*Hrj

                        So W'ir = Wir + (alpha)*(2*e*Hrj - beta*Wir)
                        and H'rj = Hrj + (alpha)*(2*e*Wir - gamma*Hrj)
                        With alpha being the rate of approximation

                        This calculation alternate between H and W evaluations at every 100 steps
                        This reduces the computing time by fixing the values of either H or W.
                        Is a cheap way of converging very fast towards the min, but it never gets quite there
                        A more complex solution like ANLS or HALS might be implemented to get activated later on and get closer to
                        the min
                    '''
                    if alternate == 1:
                        W[i][r] = abs(W[i][r] + alpha * (2 * eij * H[r][j] - beta * W[i][r]))
                    else:
                        H[r][j] = abs(H[r][j] + alpha * (2 * eij * W[i][r] - gamma * H[r][j]))
    return W, H


def cost(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=2] H, double beta, double gamma, int R):
    assert X.dtype == DTYPE and W.dtype == DTYPE and H.dtype == DTYPE

    # Cost function.
    # Calculates the total distance between W.H and X
    # it is used to break the function.
    cdef float e = 0
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] > 0:
                e = e + pow(X[i][j] - np.dot(W[i, :], H[:, j]), 2)
                for r in range(R):
                    e = e + (beta / 2) * (pow(W[i][r], 2) + (gamma / 2) * pow(H[r][j], 2))
    return e
