import numpy as np

from instance import Instance


INF = int(1e10)


def quick_preprocessing(instance: Instance):
    for n in range(instance.N):
        for k in range(instance.K):
            for m in range(instance.M):
                if instance.P[n][k][m] > instance.p:
                    instance.P[n][k][m] = INF
                    instance.R[n][k][m] = 0


def ip_dominated_processing(instance: Instance):
    for n in range(instance.N):
        arg_sorted_Pkm = np.unravel_index(np.argsort(instance.P[n], axis = None), instance.P[n].shape)
        arg_sorted_Rkm = np.unravel_index(np.argsort(instance.R[n], axis = None), instance.R[n].shape)
        rkm_list = list(zip(*arg_sorted_Rkm))
        ir = 0
        removed = set()
        for (pk, pm) in zip(*arg_sorted_Pkm):
            if (pk, pm) in removed:
                continue
            while rkm_list[ir] != (pk, pm):
                removed.add(rkm_list[ir])
                ir += 1
            if rkm_list[ir] == (pk, pm):
                ir += 1

        for (k, m) in removed:
            instance.P[n][k][m] = INF
            instance.R[n][k][m] = 0


def lp_dominated_processing(instance: Instance):
    ...
