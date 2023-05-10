"""
Plot results of overhead for Spinoza (Rust) and Spinoza Python Wrapper
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")



def read_file(filepath: str) -> list[int]:
    y = []

    with open(filepath) as f:
        for line in f:
            line = line.strip().split()
            y.extend([int(val.replace(',', '')) for val in line])

    return y


def plot_all_gates(path: str, qubit_range: tuple[int, int]) -> None:
    qubits = range(*qubit_range)
    gates = ["value_encoding"]

    for gate in gates:
        y1, y2 = [], []

        for n in qubits:
            y1.append(np.mean(read_file(path + f"/spinoza-py/{gate}-{n}qubits")))
            y2.append(np.mean(read_file(path + f"/spinoza/{gate}-{n}qubits")))

        print(np.array(y2) - np.array(y1))

        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # # fig.suptitle(f"{gate.upper()}")
        # fig.suptitle("Python Interface Overhead")

        # axes[0].plot(qubits, y1, label="Python interface", lw=0.7)
        # axes[0].plot(qubits, y2, label="Rust", lw=0.7)

        # axes[0].set_xlabel("qubits")
        # axes[0].set_ylabel("time (us)")
        # axes[0].legend(fontsize="xx-small")

        # axes[1].plot(qubits, y1, label="Python interface", lw=0.7)
        # axes[1].plot(qubits, y2, label="Rust", lw=0.7)

        # axes[1].set_yscale("log")
        # axes[1].set_xlabel("qubits")
        # axes[1].set_ylabel("time (us)")
        # axes[1].legend(fontsize="xx-small")
        # # axes[1].plot(qubits, [abs(i2 - i1) for (i1, i2) in zip(y1, y2)], lw=0.7)
        # # axes[1].set_xlabel("qubits")
        # # axes[1].set_ylabel("time difference")


        # fig.tight_layout()
        # plt.savefig(f"{gate}-gate.png", dpi=600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results", type=str, required=True)
    parser.add_argument("--start_qubits", type=int, required=True)
    parser.add_argument("--end_qubits", type=int, required=True)
    args = parser.parse_args()

    plot_all_gates(args.path_to_results, (args.start_qubits, args.end_qubits + 1))


if __name__ == "__main__":
    main()
