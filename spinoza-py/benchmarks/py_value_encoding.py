import argparse
import time
import numpy as np

from spinoza_py import QuantumCircuit, QuantumRegister, show_table

def val_encoding(n: int, v: float):
    start = time.time_ns()

    q = QuantumRegister(n)
    qc = QuantumCircuit(q)

    for i in range(n):
        qc.h(i)
    for i in range(n):
        qc.p(2 * np.pi / (2 ** (i + 1)) * v, i)

    qc.iqft(list(range(n)[::-1]))
    qc.execute()

    end = time.time_ns()
    elapsed_as_us = (end - start) // 1000
    print(elapsed_as_us)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, required=True)
    args = parser.parse_args()
    val_encoding(args.qubits, 2.4)
