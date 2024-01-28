from dataclasses import dataclass
import numpy as np

@dataclass
class Instance:
    N: int
    M: int
    K: int
    p: int

    P: np.ndarray
    R: np.ndarray


def load_from_file(filename: str) -> Instance:
    with open(filename, "r") as f:
        lines = [l[3:-1] for l in f.readlines()]

    N = int(float(lines[0]))
    M = int(float(lines[1]))
    K = int(float(lines[2]))
    p = int(float(lines[3]))

    P = np.zeros((N, K, M), dtype=int)
    for n in range(N):
        for k in range(K):
            line = lines[4 + n * K + k].split("   ")
            for (m, x) in enumerate(line):
                P[n][k][m] = int(float(x)) 

    R = np.zeros((N, K, M), dtype=int)
    for n in range(N):
        for k in range(K):
            line = lines[4 + N * K + n * K + k].split("   ")
            for (m, x) in enumerate(line):
                R[n][k][m] = int(float(x)) 

    return Instance(N, M, K, p, P, R)
