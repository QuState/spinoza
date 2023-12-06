import matplotlib.pyplot as plt
import numpy as np
from spynoza import QuantumRegister, QuantumCircuit, run

plt.style.use("ggplot")


def padded_bin(n: int, w: int) -> str:
    return bin(n)[2:].zfill(w)


def state_to_probs_dict(state):
    n = int(np.log2(len(state)))
    return dict(
        zip(
            [str(k) + "=" + padded_bin(n, k) for k in range(len(state))],
            [abs(complex(state[k][0], state[k][1])) ** 2 for k in range(len(state))],
        )
    )


def plot_probs_v_counts(counts, state_probs_dict, shots, title=""):
    bitstrings = sorted(counts.keys())
    X_axis = np.arange(len(bitstrings))
    plt.figure(figsize=(10, 5))

    plt.bar(
        X_axis - 0.2,
        [abs(v) for v in state_probs_dict.values()],
        0.4,
        align="center",
        color="grey",
        label="Expectation",
    )

    count_probabilities = [counts[bitstring] / shots for bitstring in bitstrings]
    plt.bar(
        X_axis + 0.2,
        count_probabilities,
        0.4,
        color="blue",
        label="Sample Frequency from " + str(shots) + " shots",
    )

    plt.xticks(X_axis, bitstrings, rotation=90)
    plt.xticks(rotation=90)
    plt.ylabel("Probability")
    # plt.ylim([0, 1])
    plt.title(title)
    plt.legend()
    plt.show()


def val_encoding(n, v):
    q = QuantumRegister(n)
    qc = QuantumCircuit(q)

    for i in range(n):
        qc.h(i)

    for i in range(n):
        qc.p(2 * np.pi / (2 ** (i + 1)) * v, i)

    qc.iqft(range(n)[::-1])

    return qc


if __name__ == "__main__":
    circuit = val_encoding(3, 2.4)
    state = run(circuit)
    print(len(state))
