import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

#constant declaration
L = 3 #number of non zero coef.
Tollerance = 1e-6 #tollerance for the omp algorithm

def approximation_error(Q, D):
    Q = np.array(Q)
    D = np.array(D)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=L, normalize=False)
    #omp = OrthogonalMatchingPursuit(tol=Tollerance, normalize=False)
    fitted_omp = omp.fit(D, Q)
    X1 = fitted_omp.coef_
    X1_T = np.transpose(X1)
    DX1 = np.dot(D, X1_T)
    DX1 = np.array(DX1)
    # to compute the Frobenius norm use np.linalg.norm with ord None or 'fro'
    E = np.linalg.norm((Q-DX1), ord = 'fro', keepdims=True)
    return E