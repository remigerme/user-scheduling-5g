import numpy as np

from instance import Instance


INF = int(1e10)


def preprocess(instance: Instance):
    quick_preprocessing(instance)
    ip_dominated_processing(instance)
    lp_dominated_processing(instance)


def remove_from(instance: Instance, n: int, k: int, m: int):
    instance.P[n][k][m] = INF
    instance.R[n][k][m] = 0
    instance.removed.add((n, k, m))


def quick_preprocessing(instance: Instance):
    for n in range(instance.N):
        for k in range(instance.K):
            for m in range(instance.M):
                if instance.P[n][k][m] > instance.p:
                    remove_from(instance, n, k, m)


def ip_dominated_processing(instance: Instance):
    pkm_lists = instance.get_sorted_pkm()
    for n, pkm_list in enumerate(pkm_lists):
        arg_sorted_Rkm = np.unravel_index(
            np.argsort(instance.R[n], axis=None), instance.R[n].shape
        )
        rkm_list = list(zip(*arg_sorted_Rkm))
        ir = 0
        for pk, pm in pkm_list:
            if (n, pk, pm) in instance.removed:
                continue
            while rkm_list[ir] != (pk, pm):
                remove_from(instance, n, *rkm_list[ir])
                ir += 1
            ir += 1


def lp_dominated_processing(instance: Instance):

    def test_lemma(
        n: int, a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]
    ) -> bool:
        (ak, am) = a
        (bk, bm) = b
        (ck, cm) = c
        # lemma hypothesis
        if not (instance.P[n][ak][am] < instance.P[n][bk][bm] < instance.P[n][ck][cm]):
            return False
        if not (instance.R[n][ak][am] < instance.R[n][bk][bm] < instance.R[n][ck][cm]):
            return False
        # computing both terms separately for readability
        left_term = (instance.R[n][ck][cm] - instance.R[n][bk][bm]) / (
            instance.P[n][ck][cm] - instance.P[n][bk][bm]
        )
        right_term = (instance.R[n][bk][bm] - instance.R[n][ak][am]) / (
            instance.P[n][bk][bm] - instance.P[n][ak][am]
        )
        return left_term >= right_term

    pkm_lists = instance.get_sorted_pkm()
    for n, pkm_list in enumerate(pkm_lists):
        for i in range(1, len(pkm_list) - 1):
            (pk, pm) = pkm_list[i]
            if test_lemma(n, pkm_list[i - 1], (pk, pm), pkm_list[i + 1]):
                remove_from(instance, n, pk, pm)
