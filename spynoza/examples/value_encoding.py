from math import pi
from spynoza import QuantumCircuit, QuantumRegister, show_table


def value_encoding(n, v):
    q = QuantumRegister(n)
    qc = QuantumCircuit(q)

    for i in range(n):
        qc.h(i)

    for i in range(n):
        qc.p(2 * pi / (2 ** (i + 1)) * v, i)

    qc.iqft(range(n)[::-1])

    qc.execute()
    return qc.get_statevector()


if __name__ == "__main__":
    state = value_encoding(4, 2.4)
    show_table(state)