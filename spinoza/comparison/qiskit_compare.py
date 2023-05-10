import argparse
from math import atan2, pi, sqrt

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit.visualization import plot_histogram
from sty import bg, fg


def complex_to_rgb(c, scaled_saturation=False):
    a = c.real
    b = c.imag

    val = 100

    hue = atan2(b, a) * 180 / pi
    if hue < 0:
        hue += 360

    sat = 100
    if scaled_saturation:
        sat = sqrt(a**2 + b**2) * 100

    return hsv_to_rgb(hue, sat, val)


# https://gist.github.com/eyecatchup/9536706 Colors
def hsv_to_rgb(h, s, v):
    # Make sure our arguments stay in-range
    h = max(0, min(360, h))
    s = max(0, min(100, s))
    v = max(0, min(100, v))

    # We accept saturation and value arguments from 0 to 100 because that's
    # how Photoshop represents those values. Internally, however, the
    # saturation and value are calculated from a range of 0 to 1.
    # We make that conversion here.
    s /= 100
    v /= 100

    if s == 0:
        # Achromatic (grey)
        r = g = b = v
        return [round(r * 255), round(g * 255), round(b * 255)]

    h /= 60  # sector 0 to 5
    i = np.floor(h)
    f = h - i  # factorial part of h
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    if i == 0:
        r = v
        g = t
        b = p
    elif i == 1:
        r = q
        g = v
        b = p
    elif i == 2:
        r = p
        g = v
        b = t
    elif i == 3:
        r = p
        g = q
        b = v
    elif i == 4:
        r = t
        g = p
        b = v
    else:  # case 5:
        r = v
        g = p
        b = q

    return [round(r * 255), round(g * 255), round(b * 255)]


def padded_bin(n, k):
    return bin(k)[2:].zfill(n)


def tabulate_state(state, bars=True):
    g = bg if bars else fg
    symbol = " " if bars else "#"
    n = int(np.log2(len(state)))
    from tabulate import tabulate

    reals_color = complex_to_rgb(0)
    print(
        tabulate(
            [
                [
                    str(k) + " = " + padded_bin(n, k),
                    str(np.round(state[k].real, 5)).ljust(7, "0")
                    + " + "
                    + str(np.round(state[k].imag, 5)).ljust(7, "0")
                    + "*i",
                    str(np.round(abs(state[k]), 5)).ljust(7, "0"),
                    g(*complex_to_rgb(state[k]))
                    + int(abs(state[k] * 100)) * symbol
                    + g.rs,
                    str(np.round(abs(state[k]) ** 2, 5)).ljust(7, "0"),
                    g(*reals_color) + int(abs(state[k] ** 2) * 100) * symbol + g.rs,
                ]
                for k in range(len(state))
            ],
            headers=[
                "Outcome",
                "Amplitude",
                "Magnitude",
                "Amplitude Bar",
                "Probability",
                "Probability Bar",
            ],
            tablefmt="fancy_grid",
        )
    )


def run(circuit):
    backend = qiskit.Aer.get_backend("statevector_simulator")
    job = execute(circuit, backend)
    state = job.result().get_statevector()
    # print(state)
    return state


def iqft(targets: list[int], qc):
    for j in range(len(targets)-1, -1, -1):
        qc.h(targets[j])
        for k in range(j-1, -1, -1):
            qc.cp(-np.pi / (2.0**(j - k)), targets[j], targets[k])


def gate_test(test_name, n):
    q = QuantumRegister(n)
    qc = QuantumCircuit(q)

    if test_name == "h":
        for i in range(n):
            qc.h(i)

    if test_name == "x":
        for i in range(n):
            qc.x(i)

    if test_name == "y":
        for i in range(n):
            qc.y(i)

    if test_name == "z":
        for i in range(n):
            qc.z(i)

    if test_name == "p":
        for i in range(n):
            qc.p(np.pi, i)

    if test_name == "rx":
        for i in range(n):
            qc.rx(1.0, i)

    if test_name == "ry":
        for i in range(n):
            qc.ry(1.0, i)

    if test_name == "rz":
        for i in range(n):
            qc.rz(1.0, i)

    if test_name == "u":
        for i in range(n):
            qc.u(1.0, 1.0, 1.0, i)

    if test_name == "ccx":
        for i in range(n):
            qc.h(i)
            qc.rz(3.14, i)

        qc.ccx(0, 2, 1)

    state = run(qc)

    tabulate_state(state)
    # tabulate_state(np.asarray(state)[:8])


def val_encoding(n, v):
    q0 = QuantumRegister(n)
    q1 = QuantumRegister(n)
    qc = QuantumCircuit(q0, q1)

    for i in range(n):
        qc.h(i)

    for i in range(n):
        qc.p(2 * np.pi / (2 ** (i + 1)) * v, i)

    iqft(range(n)[::-1], qc)

    state = run(qc)
    tabulate_state(np.asarray(state))


# def function_encoding(f, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", type=str, required=True)
    parser.add_argument("--qubits", type=int, required=True)

    args = parser.parse_args()

    gate_test(args.test_name, args.qubits)
