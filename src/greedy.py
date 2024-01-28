import numpy as np

from instance import Instance
from solution import Solution


def greedy(instance: Instance) -> Solution:
    X = np.zeros((instance.N, instance.K, instance.M), dtype=float)
    p_left = instance.p
    e = get_triplets_e(instance)
    ie = 0
    while p_left > 0:
        (n, k, m) = e[ie]
        p_current = min(p_left, instance.P[n][k][m])
        p_left -= p_current
        X[n][k][m] = p_current / instance.P[n][k][m]
        ie += 1

    return Solution(instance.N, instance.M, instance.K, instance.p, X)


def get_triplets_e(instance: Instance) -> list[tuple[int, int, int]]:
    """
    ...
    """
    ...