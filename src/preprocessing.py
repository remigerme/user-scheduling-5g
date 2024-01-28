import numpy as np

from instance import Instance


INF = int(1e10)


def remove_from(instance: Instance, n: int, k: int, m: int):
    instance.P[n][k][m] = INF
    instance.R[n][k][m] = 0


def quick_preprocessing(instance: Instance):
    for n in range(instance.N):
        for k in range(instance.K):
            for m in range(instance.M):
                if instance.P[n][k][m] > instance.p:
                    remove_from(instance, n, k, m)


def ip_dominated_processing(instance: Instance):
    for n in range(instance.N):
        arg_sorted_Pkm = np.unravel_index(np.argsort(instance.P[n], axis = None), instance.P[n].shape)
        arg_sorted_Rkm = np.unravel_index(np.argsort(instance.R[n], axis = None), instance.R[n].shape)
        pkm_list = list(zip(*arg_sorted_Pkm))
        rkm_list = list(zip(*arg_sorted_Rkm))
        ir = 0
        removed = set()
        for (pk, pm) in pkm_list:
            if (pk, pm) in removed:
                continue
            while rkm_list[ir] != (pk, pm):
                removed.add(rkm_list[ir])
                ir += 1
            ir += 1

        for (k, m) in removed:
            remove_from(instance, n, k, m)


def lp_dominated_processing(instance: Instance):

    def test_lemma(n: int, a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
        (ak, am) = a
        (bk, bm) = b
        (ck, cm) = c
        # lemma hypothesis
        if not (instance.P[n][ak][am] < instance.P[n][bk][bm] < instance.P[n][ck][cm]):
            return False
        if not (instance.R[n][ak][am] < instance.R[n][bk][bm] < instance.R[n][ck][cm]):
            return False
        left_term = (instance.R[n][ck][cm] - instance.R[n][bk][bm]) / (instance.P[n][ck][cm] - instance.P[n][bk][bm])
        right_term = (instance.R[n][bk][bm] - instance.R[n][ak][am]) / (instance.P[n][bk][bm] - instance.P[n][ak][am])
        return left_term >= right_term

    for n in range(instance.N):
        arg_sorted_Pkm = np.unravel_index(np.argsort(instance.P[n], axis = None), instance.P[n].shape)
        pkm_list = list(zip(*arg_sorted_Pkm))
        for i in range(1, len(pkm_list) - 1):
            (pk, pm) = pkm_list[i]
            if test_lemma(n, pkm_list[i - 1], (pk, pm), pkm_list[i + 1]):
                remove_from(instance, n, pk, pm)
