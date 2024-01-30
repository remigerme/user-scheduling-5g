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
    while p_left > 0:
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
            X[n][k][m] = (p_left - instance.P[n][k][previous_m]) / (instance.P[n][k][m] - instance.P[n][k][previous_m])
            X[n][k][previous_m] = 1 - X[n][k][m]
            p_left = 0

    return Solution(instance.N, instance.M, instance.K, instance.p, X)


def get_triplets_e(instance: Instance) -> list[tuple[int, int, int]]:
    """
    ...
    """

    def merge_sorted_lists(lists: list[list[tuple[int, int, int]]]) -> list[tuple[int, int, int]]:
        """
        Inputs : a list of N lists, each composed of L triplets
        Outputs : sorted triplets according to the corresponding values of e_l,n
        """

        def e(na, ka, ma, nb = None, kb = None, mb = None):
            if nb is None:
                return instance.R[na][ka][ma] / instance.P[na][ka][ma]
            return (instance.R[na][ka][ma] - instance.R[nb][kb][mb]) / (instance.P[na][ka][ma] - instance.P[nb][kb][mb])

        def get_maximum(lists, counters, L) -> tuple[int, tuple[int, int, int]]:
            i_max = 0
            e_max = 0
            for (n, c) in enumerate(counters):
                # Checking if counter in bounds
                if c >= L:
                    continue
                # Checking if considering useful values (and not removed by preprocessing)
                (n_, k, m) = lists[n][c]
                if instance.R[n_][k][m] == 0:
                    continue
                # Getting the minimum
                if c == 0:
                    ev = e(n_, k, m)
                else:
                    ev = e(n_, k, m, *lists[n][c - 1])
                if ev > e_max:
                    i_max = n
                    e_max = ev
            removed_values_only = e_max == 0
            return (removed_values_only, i_max, lists[i_max][counters[i_max]])

        R = []
        N = len(lists)
        L = len(lists[0])
        counters = [0] * N
        (removed_values_only, i, triplet) = get_maximum(lists, counters, L)
        while not removed_values_only:
            counters[i] += 1
            R.append(triplet)
            (removed_values_only, i, triplet) = get_maximum(lists, counters, L)
        return R

    p_lists = []
    for n in range(instance.N):
        arg_sorted_Pkm = np.unravel_index(np.argsort(instance.P[n], axis = None), instance.P[n].shape)
        n_to_zip = [n] * instance.K * instance.M # to get triplets (n, k, m) instead of (k, m) couples
        p_lists.append(list(zip(n_to_zip, *arg_sorted_Pkm)))
    E = merge_sorted_lists(p_lists)    
    return E

from instance import load_from_file
i = load_from_file("testfiles/test1.txt")
from preprocessing import *
quick_preprocessing(i)
ip_dominated_processing(i)
lp_dominated_processing(i)
print(greedy(i).X)