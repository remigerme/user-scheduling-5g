import numpy as np

from instance import Instance
from solution import Solution


def greedy(instance: Instance) -> Solution:
    X = np.zeros((instance.N, instance.K, instance.M), dtype=float)
    p_left = instance.p
    e = get_triplets_e(instance)
    ie = 0
    # We store the previously allocated m level for a given couple (N, K)
    allocated_m = np.full((instance.N, instance.K), -1, dtype=int)

    while p_left > 0 and ie < len(e):
        (n, k, m) = e[ie]
        previous_m = allocated_m[n][k]

        # Computing how much power we need to invest to go from
        # the previous m level to the new m level
        if previous_m == -1:
            effective_p = instance.P[n][k][m]
        else:
            effective_p = instance.P[n][k][m] - instance.P[n][k][previous_m]
        p_current = min(p_left, effective_p)

        # If we still have enough p_left to allocate on new m level
        # We completely allocate it on m and stop using the previous m
        # And we continue the algorithm
        if p_current == effective_p:
            p_left -= p_current
            X[n][k][previous_m] = 0
            X[n][k][m] = 1
            allocated_m[n][k] = m
            ie += 1

        # If we do not have enough power left
        # We cut the power in two so as two use everything we've got left
        # And we stop the algorithm
        else:
            X[n][k][m] = (p_left - instance.P[n][k][previous_m]) / (
                instance.P[n][k][m] - instance.P[n][k][previous_m]
            )
            X[n][k][previous_m] = 1 - X[n][k][m]
            p_left = 0

    return Solution(instance.N, instance.M, instance.K, instance.p, X)


def get_triplets_e(instance: Instance) -> list[tuple[int, int, int]]:
    """
    ...
    """

    def merge_sorted_lists(
        lists: list[list[tuple[int, int, int]]]
    ) -> list[tuple[int, int, int]]:
        """
        Inputs : a list of N lists, each composed of L triplets
        Outputs : sorted triplets according to the corresponding values of e_l,n
        """

        def e(na, ka, ma, nb=None, kb=None, mb=None):
            if nb is None:
                return instance.R[na][ka][ma] / instance.P[na][ka][ma]
            return (instance.R[na][ka][ma] - instance.R[nb][kb][mb]) / (
                instance.P[na][ka][ma] - instance.P[nb][kb][mb]
            )

        def get_maximum(lists, counters) -> tuple[int, tuple[int, int, int]]:
            i_max = 0
            e_max = -1
            for n, c in enumerate(counters):
                # Checking if counter in bounds
                if c >= len(lists[n]):
                    continue
                # Checking if considering useful values (and not removed by preprocessing)
                (n_, k, m) = lists[n][c]
                if (n_, k, m) in instance.removed:
                    continue
                # Getting the maximum
                if c == 0:
                    ev = e(n_, k, m)
                else:
                    ev = e(n_, k, m, *lists[n][c - 1])
                if ev > e_max:
                    i_max = n
                    e_max = ev
            removed_values_only = e_max == -1
            return (removed_values_only, i_max, lists[i_max][counters[i_max]])

        R = []
        counters = [0] * instance.N
        (removed_values_only, i, triplet) = get_maximum(lists, counters)
        while not removed_values_only:
            counters[i] += 1
            R.append(triplet)
            (removed_values_only, i, triplet) = get_maximum(lists, counters)
        return R

    pkm_lists = instance.get_sorted_pkm()
    # adding n to go from (k, m) to (n, k, m)
    p_lists = [
        [(n, k, m) for (k, m) in pkm_list] for (n, pkm_list) in enumerate(pkm_lists)
    ]
    E = merge_sorted_lists(p_lists)
    return E
