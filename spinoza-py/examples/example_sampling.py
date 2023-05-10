import matplotlib.pyplot as plt
import numpy as np
from spinoza_py import QuantumRegister, QuantumCircuit, run, show_table, get_samples

plt.style.use("ggplot")



def padded_bin(n, k):
    return bin(k)[2:].zfill(n)


def state_to_probs_dict(state):
    n = int(np.log2(len(state)))
    return dict(zip([str(k) + '=' + padded_bin(n, k) for k in range(len(state))],
                    [abs(complex(state[k][0], state[k][1])) ** 2 for k in range(len(state))]))


def plot_probs_v_counts(counts, state_probs_dict, shots, title=""):

    bitstrings = sorted(counts.keys())
    X_axis = np.arange(len(bitstrings))
    plt.figure(figsize=(10, 5))

    plt.bar(X_axis-0.2, [abs(v) for v in state_probs_dict.values()], 0.4, align='center', color='grey', label='Expectation')

    count_probabilities = [counts[bitstring] / shots for bitstring in bitstrings]
    plt.bar(X_axis+0.2, count_probabilities, 0.4, color='blue', label = 'Sample Frequency from ' + str(shots) + ' shots')

    plt.xticks(X_axis, bitstrings, rotation=90)
    # plt.xticks(rotation=90)
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

def normal_sin4(n, qc):
    theta = np.arccos(np.sqrt(2 / 3))  # 0.47
    # qc = QuantumCircuit(q)
    # n = q.len()
    # print(n)
    qc.ry(2 * theta, n - 1)
    qc.p(np.pi, n - 1)
    qc.cry(np.pi / 2, n - 1, 0)

    for i in range(1, n - 1):
        qc.cx(0, i)


def test_ccx(n, v):

    q0 = QuantumRegister(n)
    qc = QuantumCircuit(q0)

    for i in range(n):
        qc.h(i)

    for i in range(n):
        qc.p(2 * np.pi / (2 ** (i + 1)) * v, i)

    return qc


def linear_1_plus_2k_inv(qc):
    # qc = QuantumCircuit(q)
    qc.cry(-8.497482742761816, 0, 1)
    qc.ry(-5.548837639543121, 2)
    qc.cry(-3.533015961552675, 2, 0)
    qc.cry(-2.498091544783385, 0, 1)
    qc.ry(-2.9253622881159256, 2)
    qc.ry(-1.2309594173416536, 0)

    return qc


def normal_4_genetic(qc):
    # qc = QuantumCircuit(q)
    qc.ry(1.493677152222903, 3)
    qc.cry(9.918996489194324, 3, 2)
    qc.ry(4.253218556940338, 0)
    qc.ry(8.880463706841903, 2)
    qc.cry(5.073159678386604, 3, 1)
    qc.ry(3.024323083524102, 3)
    qc.cry(5.108152604667179, 3, 1)
    qc.cry(6.573305373762854, 0, 3)
    qc.cry(3.1447987386882286, 1, 0)
    qc.ry(0.44596209419350297, 1)
    qc.cry(5.206069555962428, 3, 0)
    qc.x(2)
    qc.cry(0.15193053179197832, 2, 0)
    qc.cry(2.24645325201094, 0, 1)
    return qc


def build_dagger(n, operator_a, operator_b):
    q = QuantumRegister(n)
    qc = QuantumCircuit(q)
    qc = operator_a(n, qc)
    qc = operator_b(qc)
    return qc


# inner product
# circuit = build_dagger(3, normal_sin4, linear_1_plus_2k_inv)
# state = run(circuit)
# print(state)

circuit = val_encoding(3, 2.4)
state = run(circuit)
print(len(state))

# sample
# samples = get_samples(state, 1000)
#
# plot_probs_v_counts(samples, state_to_probs_dict(state), 1000)
