import numpy as np

from instance import Instance
from solution import Solution


def greedy(instance: Instance) -> Solution:
    X = np.zeros((instance.N, instance.K, instance.M), dtype=float)
    ...
    return Solution(instance.N, instance.M, instance.K, instance.p, X)
