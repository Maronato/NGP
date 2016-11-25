import numpy as np

cimport numpy as np

DTYPE = np.double

ctypedef np.double_t DTYPE_t


def ALS(self, np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=2] H):

    assert X.dtype == DTYPE and W.dtype == DTYPE and H.dtype == DTYPE
    cdef float eij
    # alternate between H and W evaluations at every step
    # This reduces the computing time by fixing the values of either H or W.
    # Is a cheap way of converging very fast towards the min, but it never gets quite there
    # Also called ALS(https://arxiv.org/pdf/1401.5226v1.pdf)
    # A more complex solution like ANLS or HALS might be implemented to get activated later on and get closer to
    # the min
    if self.alternate == 1:
        self.alternate = 0
    else:
        self.alternate = 1
    for i in range(len(X)):
        for j in range(len(X[i])):

            # Only evaluate WH if the current grade is set
            if X[i][j] > 0:
                # Distance between X and WH
                eij = X[i][j] - np.dot(W[i, :], H[:, j])
                # Not very efficient way of minimizing the erimror
                # reduce the error for every feature
                for r in range(self.R):
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
                        '''
                    if self.alternate == 1:
                        W[i][r] = abs(W[i][r] + self.alpha * (2 * eij * H[r][j] - self.beta * W[i][r]))
                    else:
                        H[r][j] = abs(H[r][j] + self.alpha * (2 * eij * W[i][r] - self.gamma * H[r][j]))
    return W, H
