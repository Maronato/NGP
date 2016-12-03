import numpy as np
# usage from:
# http://docs.cython.org/en/latest/src/tutorial/numpy.html

cimport numpy as np

DTYPE = np.double

ctypedef np.double_t DTYPE_t


def AU(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=2] H, double alpha, double beta, double gamma, int alternate, int R):
    """Additive Update.

    Regular AU calculations, component-wise.
    More info: https://gradeprocessing.herokuapp.com/info/
    """
    assert X.dtype == DTYPE and W.dtype == DTYPE and H.dtype == DTYPE
    cdef float eij
    for i in range(len(X)):
        for j in range(len(X[i])):

            # Only evaluate W, H if the current grade is set
            if X[i][j] > 0:
                # Distance between X and V
                eij = X[i][j] - np.dot(W[i, :], H[:, j])
                for r in range(R):

                    # Switch between W and H
                    if alternate == 1:
                        W[i][r] = abs(W[i][r] + alpha * (2 * eij * H[r][j] - beta * W[i][r]))
                    else:
                        H[r][j] = abs(H[r][j] + alpha * (2 * eij * W[i][r] - gamma * H[r][j]))
    return W, H


def MU(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=2] H, int alternate, int R):
    """Multiplicative Update.

    Regular MU calculations, component-wise.
    More info: https://gradeprocessing.herokuapp.com/info/
    """
    assert X.dtype == DTYPE and W.dtype == DTYPE and H.dtype == DTYPE

    for i in range(len(X)):
        for j in range(len(X[i])):
            # Only evaluate W, H if the current grade is set
            if X[i][j] > 0:
                for r in range(R):

                    # Switch between W and H using alternate
                    if alternate == 1:
                        W[i][r] = W[i][r] * ((X.dot(H.T))[i][r]) / ((W.dot(H.dot(H.T)))[i][r])
                    else:
                        H[r][j] = H[r][j] * ((W.T.dot(X))[r][j]) / ((W.T.dot(W).dot(H))[r][j])
    return W, H


def cost(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=2] H, double beta, double gamma, int R):
    assert X.dtype == DTYPE and W.dtype == DTYPE and H.dtype == DTYPE
    """Cost function.

    Calculates the total distance between W.H and X
    it is used to break the function.
    More info: https://gradeprocessing.herokuapp.com/info/
    """
    cdef float e = 0
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] > 0:
                e = e + pow(X[i][j] - np.dot(W[i, :], H[:, j]), 2)
                for r in range(R):
                    e = e + (beta / 2) * (pow(W[i][r], 2) + (gamma / 2) * pow(H[r][j], 2))
    return e
