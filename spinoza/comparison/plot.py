"""
Plot results of benchmarks for Qulacs C++ and Spinoza (Rust)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import rsmf


def read_file(filepath: str) -> list[int]:
    y = []

    with open(filepath) as f:
        for line in f:
            line = line.strip().split()
            y.extend([int(val.replace(",", "")) for val in line])

    return y


def ratio_plot() -> None:
    fig, axs = plt.subplots(8, figsize=(7, 20))
    fig.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5
    )
    fig.suptitle("Rust vs. Qulacs C++")

    for i in range(1, 9):
        x1, y1 = read_file(f"/tmp/cpp-qulacs-benchmark-{i}threads")
        x2, y2 = read_file(f"/tmp/rust-quantum-benchmark-{i}threads")
        x1, y1 = x1[16:], y1[16:]
        x2, y2 = x2[16:], y2[16:]
        y = [y_i / y_j for (y_i, y_j) in zip(y2, y1)]
        axs[i - 1].plot(x1, y, label=f"{i}-threads")
        axs[i - 1].set_xlabel("qubits", fontsize=10)
        axs[i - 1].set_ylabel("ratio", fontsize=10)
        axs[i - 1].set_title(f"{i} threads", fontsize=10)

    # plt.xlabel("qubits")
    # plt.ylabel("ratio")
    # plt.legend(fontsize="xx-small")
    # plt.title("Rust vs. Qulacs C++ w/ 8 threads")
    plt.savefig("ratio-plot.png", dpi=600)


def plot_all_gates(path: str, qubit_range: tuple[int, int], latex=False) -> None:
    qubits = range(*qubit_range)
    gates = ("rz", "rx", "ry", "h", "z", "p", "x", "value_encoding", "qcbm")
    i = 0

    for gate in gates:
        y1, y2 = [], []

        for n in qubits:
            y1.append(np.mean(read_file(path + f"/qulacs/{gate}-{n}qubits")))
            y2.append(np.mean(read_file(path + f"/spinoza/{gate}-{n}qubits")))

        if latex:
            fig = fmt.figure()
            fig_filename = f"{gate}-gate.pgf"
        else:
            plt.figure(i)
            i += 1
            fig_filename = f"{gate}-gate.png"

        plt.plot(qubits, y1, label="qulacs", lw=0.7)
        plt.plot(qubits, y2, label="spinoza", lw=0.7)

        plt.xlabel("qubits")
        plt.ylabel("time (us)")
        plt.legend(fontsize="xx-small")
        plt.tight_layout(pad=0.0)

        if latex:
            plt.savefig(fig_filename)
        else:
            plt.savefig(fig_filename, dpi=600)

        # next plot -- log-linear
        if latex:
            fig = fmt.figure()
            fig_filename = f"{gate}-gate-log-linear.pgf"
        else:
            plt.figure(i)
            i += 1
            fig_filename = f"{gate}-gate-log-linear.png"

        plt.plot(qubits, y1, label="qulacs", lw=0.7)
        y2_adjust = [0.49 if x == 0 else x for x in y2]
        plt.plot(qubits, y2_adjust, label="spinoza", lw=0.7)

        plt.yscale("log")
        plt.xlabel("qubits")
        plt.ylabel("time (us)")
        plt.legend(fontsize="xx-small")
        plt.tight_layout(pad=0.0)

        if latex:
            plt.savefig(fig_filename)
        else:
            plt.savefig(fig_filename, dpi=600)


if __name__ == "__main__":
    plt.style.use("ggplot")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results", type=str, required=True)
    parser.add_argument("--start_qubits", type=int, required=True)
    parser.add_argument("--end_qubits", type=int, required=True)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    if args.latex:
        fmt = rsmf.setup("../../paper/rust-sim-paper.tex")

    plot_all_gates(args.path_to_results, (args.start_qubits, args.end_qubits + 1), latex=args.latex)
