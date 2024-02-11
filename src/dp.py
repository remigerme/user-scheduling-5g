import numpy as np

from instance import Instance
from solution import Solution


"""
Note : despite our best efforts, these DP do not work :(
"""


def dp(instance: Instance) -> Solution:
    p_max, N, K, M = instance.p, instance.N, instance.K, instance.M

    # First step : compute max data rate of each subproblem
    # We consider subproblems where we allow
    # the use of channels <= n and a power <= p (both 1 indexed)
    pkm_lists = instance.get_sorted_pkm()
    D = np.zeros((N + 1, p_max + 1), dtype=int)
    for n, pkm_list in enumerate(pkm_lists):
        for p in range(1, p_max + 1):
            for k, m in pkm_list:
                if p >= instance.P[n][k][m]:
                    D[n + 1][p] = max(
                        D[n][p],
                        D[n + 1][p],
                        instance.R[n][k][m] + D[n][p - instance.P[n][k][m]],
                    )
                # Once one Pnkm is too big, all the next ones are too
                # Cause they are sorted in increasing order
                else:
                    break

    # Second step : we determine the optimal solution
    # from the previously computed array of subproblems
    X = np.zeros((N, K, M), dtype=int)
    p = p_max
    previous_p = p_max + 1

    while p != previous_p:
        previous_p = p
        for n, pkm_list in enumerate(pkm_lists):
            # Channel n is not allocated
            if D[n + 1][p] == D[n][p]:
                continue
            for k, m in pkm_list:
                # (k, m) is the best couple for channel n
                if D[n + 1][p] == instance.R[n][k][m] + D[n][p - instance.P[n][k][m]]:
                    # not considering intermediate subproblem solutions
                    X[n][k][m] = 1
                    p -= instance.P[n][k][m]
                    break

    return Solution(N, M, K, p_max, X)


def dp_bis(instance: Instance, U: int) -> Solution:
    p_max, N, K, M = instance.p, instance.N, instance.K, instance.M

    # First step : compute min power for each subproblem
    # We consider subproblems where we allow
    # the use of channels <= n and a data rate <= r (both 1 indexed)
    pkm_lists = instance.get_sorted_pkm()
    INF = p_max + 1
    D = np.full((N + 1, U + 1), INF, dtype=int)
    for n in range(N + 1):
        D[n][0] = 0
    for n, pkm_list in enumerate(pkm_lists):
        for r in range(1, U + 1):
            D[n + 1][r] = D[n][r]
            for k, m in pkm_list:
                if r >= instance.R[n][k][m]:
                    D[n + 1][r] = min(
                        D[n + 1][r],
                        D[n][r - instance.R[n][k][m]] + instance.P[n][k][m],
                    )
    return D[N][U]
