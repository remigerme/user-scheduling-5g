from matplotlib.pyplot import plot, legend, show, xticks, title

from instance import Instance, load_from_file
from preprocessing import quick_preprocessing, ip_dominated_processing, lp_dominated_processing


def count_triplets(instance: Instance) -> int:
    s = 0
    for n in range(instance.N):
        for k in range(instance.K):
            for m in range(instance.M):
                if instance.R[n][k][m] != 0:
                    s += 1
    return s


def show_graph():
    categories = [
        "Initial",
        "After quick processing",
        "After removing \nIP-dominated terms",
        "After removing \nLP-dominated terms"
    ]
    x_positions = range(len(categories))
    for n in range(5):
        i = load_from_file(f"testfiles/test{n + 1}.txt")
        tot = i.N * i.M * i.K
        y = [count_triplets(i) / tot]
        quick_preprocessing(i)
        y.append(count_triplets(i) / tot)
        ip_dominated_processing(i)
        y.append(count_triplets(i) / tot)
        lp_dominated_processing(i)
        y.append(count_triplets(i) / tot)
        plot(x_positions, y, label=f"Test {n + 1}")
    xticks(x_positions, categories)
    title("Ratio evolution of triplets count after each step of preprocessing")
    legend()
    show()


if __name__ == "__main__":
    show_graph()
