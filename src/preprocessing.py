import numpy as np

from instance import Instance


INF = int(1e10)


def quick_preprocessing(instance: Instance):
    for n in range(instance.N):
        for m in range(instance.M):
            for k in range(instance.K):
                if instance.P[n][m][k] > instance.p:
                    instance.P[n][m][k] = INF
                    instance.R[n][m][k] = 0


def ip_dominated_processing(instance: Instance):
    ...

def lp_dominated_processing(instance: Instance):
    ...
