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
print(i.P)
print(i.R)
print(get_triplets_e(i))