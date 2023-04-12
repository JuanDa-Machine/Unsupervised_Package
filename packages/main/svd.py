import numpy as np
def svd(X):
    U, s, V = np.linalg.svd(X, full_matrices=False)
    return U, s, V