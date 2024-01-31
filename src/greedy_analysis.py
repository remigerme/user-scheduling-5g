import numpy as np
from scipy.optimize import linprog, OptimizeResult

from instance import Instance, load_from_file
from solution import Solution
from preprocessing import preprocess
from greedy import greedy

def get_c(instance: Instance) -> np.ndarray:
    """
    Use this function without preprocessing !
    """
    R = np.copy(instance.R)
    # Incremental version of R
    for n in range(instance.N):
        for k in range(instance.K):
            for m in range(instance.M - 1, 0, -1):
                R[n][k][m] = R[n][k][m] - R[n][k][m - 1]
    return np.ndarray.flatten(R)

def get_A(instance: Instance) -> np.ndarray:
    """
    Use this function without preprocessing !
    """
    P = np.copy(instance.P)
    # Incremental version of P
    for n in range(instance.N):
        for k in range(instance.K):
            for m in range(instance.M - 1, 0, -1):
                P[n][k][m] = P[n][k][m] - P[n][k][m - 1]
    A_vector = np.ndarray.flatten(P)
    return np.reshape(A_vector, (1, instance.N * instance.K * instance.M))

def solve(instance: Instance) -> OptimizeResult:
    """
    Use this function without preprocessing !
    """
    c = get_c(instance)
    A = get_A(instance)
    b = np.full(1, instance.p, dtype=int)
    # Careful : linprog minimizes, we want to maximize
    return linprog(- c, A_ub=A, b_ub=b, bounds=(0, 1))
