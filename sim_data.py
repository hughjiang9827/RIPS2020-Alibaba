import numpy as np

def sim_data(n, m, k, l, c_bound=10, M_bound=10, A_bound=10, seed=42):
    np.random.seed(seed)

    c = np.random.uniform(c_bound, size=(n, m))
    M = np.random.uniform(M_bound, size=(n, k, m))
    # TODO: check bound
    d = np.random.uniform(n * m * M_bound / 4, n * m * M_bound * 3 / 4, size=(k))
    A = np.random.uniform(A_bound, size=(n, l, m))
    b = np.random.uniform(m * A_bound / 4, m * A_bound * 3 / 4, size=(n, l))
    return c, M, A, d, b
