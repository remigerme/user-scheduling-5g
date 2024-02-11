from time import time

import numpy as np
from scipy.optimize import linprog, OptimizeResult
import matplotlib.pyplot as plt

from instance import Instance, load_from_file
from solution import Solution
from preprocessing import preprocess
from greedy import greedy


def get_c(instance: Instance) -> np.ndarray:
    return np.ndarray.flatten(instance.R)


def get_A(instance: Instance) -> np.ndarray:
    N, K, M = instance.N, instance.K, instance.M
    A_cost = np.ndarray.flatten(instance.P)
    A_channel = np.zeros(N * N * K * M)
    for n in range(N):
        A_channel[(n * N + n) * K * M : (n * N + n + 1) * K * M] = 1
    A = np.concatenate((A_cost, A_channel))
    return np.reshape(A, (N + 1, N * K * M))


def solve(instance: Instance) -> OptimizeResult:
    c = get_c(instance)
    A = get_A(instance)
    b = np.full(instance.N + 1, 1, dtype=int)
    b[0] = instance.p
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
    scale = [50, 15]
    fig, (rates, times) = plt.subplots(1, 2)
    categories = ["Greedy", "LP solver"]
    x_positions = range(len(categories))
    for n in range(5):
        i = load_from_file(f"testfiles/test{n + 1}.txt")
        lp_start = time()
        solved = solve(i)
        lp_duration = time() - lp_start
        preprocess(i)
        greedy_start = time()
        s = greedy(i)
        greedy_duration = time() - greedy_start
        abs = [power_used(i, s) / i.p, 1 - solved.slack[0] / i.p]
        ords = [data_rate(i, s), -solved.fun]
        rates.scatter(abs, ords, s=scale, label=f"Test {n + 1}", edgecolors="black")
        if n >= 3:
            times.scatter(
                x_positions, [greedy_duration, lp_duration], label=f"Test {n + 1}"
            )
        print(
            (
                f"Test {n + 1} :\n"
                f"Power usage (greedy, lp): {abs}\n"
                f"Data rates (idem) : {ords}\n"
                f"CPU time (s) (idem) : {greedy_duration}, {lp_duration}\n"
            )
        )

    # Plotting rates
    rates.set_title("Greedy (wide dots) vs\nScipy LP solver (small dots)")
    rates.set_yscale("log")
    rates.set_xlabel("Power used / Total available power")
    rates.set_ylabel("Total data rate")
    rates.legend()

    # Plotting times
    times.set_title("CPU time")
    times.set_xticks(x_positions, categories)
    times.set_yscale("log")
    times.set_ylabel("Time elapsed (s)")
    times.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_graph()
