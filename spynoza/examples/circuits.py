import argparse

import numpy as np
from spynoza import QuantumRegister, QuantumCircuit


def qcbm(n):
    pairs = [(i, (i + 1) % n) for i in range(n)]
    q = QuantumRegister(n)
    qc = QuantumCircuit(q)

    for i in range(n):
        qc.rx(1.0, i)
        qc.rz(1.0, i)

    for a, b in pairs:
        qc.cx(a, b)

    for d in range(9):
        for i in range(n):
            qc.rz(1.0, i)
            qc.rx(1.0, i)
            qc.rz(1.0, i)

        for a, b in pairs:
            qc.cx(a, b)

    for i in range(n):
        qc.rz(1.0, i)
        qc.rx(1.0, i)

    qc.execute()


def x_gate(n):
    q = QuantumRegister(n)
    qc = QuantumCircuit([q])

    for i in range(n):
        qc.x(i)

    qc.execute()


def h_gate(n):
    q = QuantumRegister(n)
    qc = QuantumCircuit([q])

    for i in range(n):
        qc.h(i)

    qc.execute()


def rx_gate(n):
    q = QuantumRegister(n)
    qc = QuantumCircuit([q])

    for i in range(n):
        qc.rx(np.pi / 4, i)

    qc.execute()


def rz_gate(n):
    q = QuantumRegister(n)
    qc = QuantumCircuit([q])

    for i in range(n):
        qc.rz(np.pi / 4, i)

    qc.execute()


def ry_gate(n):
    q = QuantumRegister(n)
    qc = QuantumCircuit([q])

    for i in range(n):
        qc.ry(np.pi / 4, i)

    qc.execute()


def phase_gate(n):
    q = QuantumRegister(n)
    qc = QuantumCircuit([q])

    for i in range(n):
        qc.p(np.pi / 4, i)

    qc.execute()


def cx_gate(n):
    pairs = [(i, (i + 1) % n) for i in range(n)]
    q = QuantumRegister(n)
    qc = QuantumCircuit([q])

    for a, b in pairs:
        qc.cx(a, b)

    qc.execute()


def run(n: int, name: str):
    if name == "qcbm":
        qcbm(n)
    if name == "x_gate":
        x_gate(n)
    if name == "rx_gate":
        rx_gate(n)
    if name == "rz_gate":
        rz_gate(n)
    if name == "cnot_gate":
        cx_gate(n)
    if name == "phase_gate":
        phase_gate(n)
    if name == "h_gate":
        h_gate(n)
    if name == "ry_gate":
        ry_gate(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    run(args.qubits, args.name)
