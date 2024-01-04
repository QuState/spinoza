import random
from math import pi, atan

from spynoza import QuantumCircuit, QuantumRegister, show_table


def quantum_lstm():
    n = 4
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    x_params = [random.uniform(-10, 10) for _ in range(n)]
    alphas_betas_gammas = [[random.uniform(0, 2.0 * pi) for _ in range(3)] for _ in range(n)]

    for target in range(n):
        qc.h(target)

    for target in range(n):
        x = x_params[target]
        qc.ry(atan(x), target)

    for target in range(n):
        x = x_params[target]
        qc.rz(atan(x * x), target)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 0)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(2, 0)
    qc.cx(3, 1)

    for target in range(n):
        alpha, beta, gamma = alphas_betas_gammas[target]
        qc.u(alpha, beta, gamma, target)

    qc.execute()
    return qc.state_vector


if __name__ == "__main__":
    state = quantum_lstm()
    print(show_table(state))
