import numpy as np

from instance import Instance
from solution import Solution


def dp(instance: Instance) -> Solution:
    p_max, N, K, M = instance.p, instance.N, instance.K, instance.M
    D = np.zeros((p_max + 1, N + 1, K + 1, M + 1), dtype=int)
    PM = np.zeros((p_max + 1, N + 1, K + 1), dtype=int)
    for p in range(1, p_max + 1):
        for n in range(1, N + 1):
            D[p][n][0][M] = D[p][n - 1][K][M]
            for k in range(1, K + 1):
                D[p][n][k][0] = D[p][n][k - 1][M]
                for m in range(1, M + 1):
                    previous_value = D[p][n][k][m - 1]
                    current_cost = instance.P[n - 1][k - 1][m - 1]
                    if current_cost <= p:
                        increment_value = instance.R[n - 1][k - 1][m - 1]
                        if PM[p][n][k] > 0:
                            increment_value -= instance.R[n - 1][k - 1][PM[p][n][k] - 1]

                        current_value = D[p - current_cost][N][K][M] + increment_value
                        if current_value >= previous_value:
                            D[p][n][k][m] = current_value
                            PM[p][n][k] = m
                    else:
                        # No need to update PM[p][n][k]
                        D[p][n][k][m] = previous_value
