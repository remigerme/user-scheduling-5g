import numpy as np

from instance import Instance
from solution import Solution


triplet = tuple[int, int, int]


def get_maximum(
    instance: Instance,
    lists: list[list[triplet]],
    counters: list[int],
) -> triplet:
    """
    Return the triplet which maximizes e which has not
    been allocated yet (and not removed by preprocessing)
    """

    def e(n, ka, ma, kb=None, mb=None):
        if kb is None:
            return instance.R[n][ka][ma] / instance.P[n][ka][ma]
        return (instance.R[n][ka][ma] - instance.R[n][kb][mb]) / (
            instance.P[n][ka][ma] - instance.P[n][kb][mb]
        )

    n_max = 0
    e_max = -1
    for n, c in enumerate(counters):
        # Checking if counter in bounds
        if c >= len(lists[n]):
            continue
        # Checking if considering useful values (and not removed by preprocessing)
        k, m = lists[n][c]
        if (n, k, m) in instance.removed:
            continue
        # Getting the maximum
        if c == 0:
            ev = e(n, k, m)
        else:
            ev = e(n, k, m, *lists[n][c - 1])
        if ev > e_max:
            n_max = n
            e_max = ev
    removed_values_only = e_max == -1
    if removed_values_only:
        return (True, (None, None, None))
    return (False, (n_max, *lists[n_max][counters[n_max]]))


def greedy(instance: Instance) -> Solution:
    X = np.zeros((instance.N, instance.K, instance.M), dtype=float)
    p_lists = instance.get_sorted_pkm()
    p_left = instance.p
    # For each channel, we store the couple (k, m) currently allocated
    allocations = [(-1, -1)] * instance.N
    # We iterate simultaneously over the N lists so we need N counters
    counters = [0] * instance.N

    (removed_values_only, (n, k, m)) = get_maximum(instance, p_lists, counters)

    while p_left > 0 and not removed_values_only:
        previous_k, previous_m = allocations[n]
        # Computing how much power we need to invest to go from
        # the previous (k, m) allocation to the new one
        if (previous_k, previous_m) == (-1, -1):
            effective_p = instance.P[n][k][m]
        else:
            effective_p = instance.P[n][k][m] - instance.P[n][previous_k][previous_m]

        # If we still have enough p_left to allocate on new m level
        # We completely allocate it on m and stop using the previous m
        # And we continue the algorithm
        if p_left >= effective_p:
            p_left -= effective_p
            X[n][previous_k][previous_m] = 0
            X[n][k][m] = 1
            allocations[n] = (k, m)
            counters[n] += 1
            (removed_values_only, (n, k, m)) = get_maximum(instance, p_lists, counters)

        # If we do not have enough power left
        # We cut the power in two so as two use everything we've got left
        # And we stop the algorithm
        else:
            if (previous_k, previous_m) == (-1, -1):
                X[k][n][m] = p_left / instance.P[n][k][m]
            else:
                X[n][k][m] = (p_left - instance.P[n][previous_k][previous_m]) / (
                    instance.P[n][k][m] - instance.P[n][previous_k][previous_m]
                )
                X[n][previous_k][previous_m] = 1 - X[n][k][m]
            p_left = 0

    return Solution(instance.N, instance.M, instance.K, instance.p, X)
