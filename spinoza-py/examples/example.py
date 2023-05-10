import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from spinoza_py import QuantumRegister, QuantumCircuit, show_table

plt.style.use("ggplot")


def val_encoding(n, v):
    q = QuantumRegister(n)
    qc = QuantumCircuit(q)

    for i in range(n):
        qc.h(i)

    for i in range(n):
        qc.p(2 * np.pi / (2 ** (i + 1)) * v, i)

    qc.iqft(range(n)[::-1])

    qc.execute()


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


def time_benchmark(
    n, num_runs, test_name
):

    run_times = [0] * num_runs

    for i in range(num_runs):
        if test_name == "value_encoding":
            now = time.time()
            val_encoding(n, 7.0)
            end = time.time()
        if test_name == "qcbm":
            now = time.time()
            qcbm(n)
            end = time.time()
        if test_name == "x_gate":
            now = time.time()
            x_gate(n)
            end = time.time()
        if test_name == "rx_gate":
            now = time.time()
            rx_gate(n)
            end = time.time()
        if test_name == "rz_gate":
            now = time.time()
            rz_gate(n)
            end = time.time()
        if test_name == "cnot_gate":
            now = time.time()
            cx_gate(n)
            end = time.time()
        if test_name == "phase_gate":
            now = time.time()
            phase_gate(n)
            end = time.time()
        if test_name == "h_gate":
            now = time.time()
            h_gate(n)
            end = time.time()
        if test_name == "ry_gate":
            now = time.time()
            ry_gate(n)
            end = time.time()

        elapsed = (end - now) * 10**6
        run_times[i] = elapsed

    print(np.mean(run_times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--num_runs", type=int, required=True)
    parser.add_argument("--test_name", type=str, required=True)
    args = parser.parse_args()

    time_benchmark(args.qubits, args.num_runs, args.test_name)
