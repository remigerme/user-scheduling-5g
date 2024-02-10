import numpy as np
from scipy.optimize import linprog, OptimizeResult
from matplotlib.pyplot import title, legend, show, scatter, xlim, xlabel, ylabel, yscale

from instance import Instance, load_from_file
from solution import Solution
from preprocessing import preprocess
from greedy import greedy


def get_c(instance: Instance) -> np.ndarray:
    """
    Use this function without preprocessing !
    """
    R = np.copy(instance.R)
    # Incremental version of R
    for n in range(instance.N):
        for k in range(instance.K):
            for m in range(instance.M - 1, 0, -1):
                R[n][k][m] = R[n][k][m] - R[n][k][m - 1]
    return np.ndarray.flatten(R)


def get_A(instance: Instance) -> np.ndarray:
    """
    Use this function without preprocessing !
    """
    P = np.copy(instance.P)
    # Incremental version of P
    for n in range(instance.N):
        for k in range(instance.K):
            for m in range(instance.M - 1, 0, -1):
                P[n][k][m] = P[n][k][m] - P[n][k][m - 1]
    A_vector = np.ndarray.flatten(P)
    return np.reshape(A_vector, (1, instance.N * instance.K * instance.M))


def solve(instance: Instance) -> OptimizeResult:
    """
    Use this function without preprocessing !
    """
    c = get_c(instance)
    A = get_A(instance)
    b = np.full(1, instance.p, dtype=int)
    # Careful : linprog minimizes, we want to maximize
    return linprog(-c, A_ub=A, b_ub=b, bounds=(0, 1))


def data_rate(instance: Instance, solution: Solution) -> float:
    s = 0
    for n in range(instance.N):
        for k in range(instance.K):
            for m in range(instance.M):
                s += solution.X[n][k][m] * instance.R[n][k][m]
    return s


def power_used(instance: Instance, solution: Solution) -> float:
    p = 0
    for n in range(instance.N):
        for k in range(instance.K):
            for m in range(instance.M):
                p += solution.X[n][k][m] * instance.P[n][k][m]
    return p


def show_graph():
    scale = [30, 10]
    for n in range(5):
        i = load_from_file(f"testfiles/test{n + 1}.txt")
        solved = solve(i)
        preprocess(i)
        s = greedy(i)
        abs = [power_used(i, s) / i.p, 1 - solved.slack[0] / i.p]
        ords = [data_rate(i, s), -solved.fun]
        print(n, abs, ords)
        scatter(abs, ords, s=scale, label=f"Test {n + 1}")
    title("")
    xlim(0.9, xlim()[1])
    yscale("log")
    xlabel("Power used / Total available power")
    ylabel("Total data rate")
    legend()
    show()


if __name__ == "__main__":
    show_graph()
