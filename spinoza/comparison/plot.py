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


def plot_gate(path: str, qubit_range: tuple[int, int], gate: str, latex=False) -> None:
    qubits = range(*qubit_range)
    i = 0

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
    parser.add_argument("--gate", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    if args.latex:
        fmt = rsmf.setup("../../paper/rust-sim-paper.tex")

    plot_gate(args.path_to_results, (args.start_qubits, args.end_qubits + 1), args.gate, latex=args.latex)
