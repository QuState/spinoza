from math import pi

from qiskit import (Aer, QuantumCircuit as QC, QuantumRegister as QR)
from spynoza import QuantumRegister, QuantumCircuit, PyQuantumTransformation

# Run the quantum circuit on a statevector simulator backend
backend = Aer.get_backend('statevector_simulator')


def iqft_qiskit(n: int, targets: list[int]) -> QC:
    circuit = QC(n)

    for j in reversed(range(len(targets))):
        circuit.h(targets[j])
        for k in reversed(range(j)):
            circuit.cp((-pi / (j - k) ** 2), targets[j], targets[k])

    return circuit


def iqft_spynoza(n: int, targets: list[int]) -> QuantumCircuit:
    qr = QuantumRegister(n)
    circuit = QuantumCircuit(qr)

    for j in reversed(range(len(targets))):
        circuit.h(targets[j])
        for k in reversed(range(j)):
            circuit.cp((-pi / (j - k) ** 2), targets[j], targets[k])

    return circuit


def simple_spynoza(n: int) -> QuantumCircuit:
    qr = QuantumRegister(n)
    circuit = QuantumCircuit(qr)

    circuit.h(0)
    circuit.x(1)
    return circuit


def all_gates_spynoza() -> QuantumCircuit:
    n = 7
    qr = QuantumRegister(n)
    circuit = QuantumCircuit(qr)

    for t in range(n):
        circuit.h(t)

    t = 0
    circuit.x(t)
    t += 1
    circuit.y(t)
    t += 1
    circuit.z(t)
    t += 1
    circuit.p(1.457, t)
    t += 1
    circuit.rx(2.457, t)
    t += 1
    circuit.rx(3.457, t)
    t += 1
    circuit.rz(4.11, t)

    for t in range(n):
        circuit.u(1.29, 2.58, 3.67, t)

    t = 1
    circuit.cx(t - 1, t)
    t += 1
    circuit.cy(t - 1, t)
    t += 1
    # circuit.cz(t-1, t)
    circuit.cp(1.46, t - 1, t)
    t += 1
    circuit.crx(2.46, t - 1, t)
    t += 1
    circuit.cry(3.46, t - 1, t)
    t += 1
    # circuit.crz(4.11, t-1, t)

    return circuit


def get_jump_table(qc):
    """Generate jump tables for a Quantum Circuit"""
    jp_no_control_no_arg = {'h': qc.h, 'x': qc.x, 'y': qc.y, 'z': qc.z}
    jp_no_control_single_arg = {'p': qc.p, 'rx': qc.rx, 'ry': qc.ry, 'rz': qc.rz}
    jp_no_control_multi_arg = {'u': qc.u}
    jp_sc_no_arg = {'h': qc.ch, 'x': qc.cx, 'y': qc.cy, 'z': qc.cz}
    jp_sc_single_arg = {'p': qc.cp, 'rx': qc.crx, 'ry': qc.cry, 'rz': qc.crz}
    jp_sc_multi_arg = {'u': qc.cu}
    # jump_table_mc = {}
    return jp_no_control_no_arg, jp_no_control_single_arg, jp_no_control_multi_arg, jp_sc_no_arg, jp_sc_single_arg, jp_sc_multi_arg


def spynoza_to_qiskit(register_sizes: list[int], transformations: list[PyQuantumTransformation]) -> QC:
    """Convert a list of Spynoza transformatons into a Qiskit circuit"""
    qs = [QR(size=size) for size in register_sizes]
    qc = QC(*qs)

    jp_no_control_no_arg, jp_no_control_single_arg, jp_no_control_multi_arg, jp_sc_no_arg, jp_sc_single_arg, jp_sc_multi_arg = get_jump_table(
        qc)

    for tr in transformations:

        # no controls and no args ==> 00
        if tr.controls is None and tr.arg is None:
            try:
                jp_no_control_no_arg[tr.name](tr.target)
            except KeyError:
                raise f"{tr.name} not found in jump table"
        # no controls and args ==> 01
        elif tr.controls is None and tr.arg:
            if tr.name in jp_sc_single_arg:
                jp_no_control_single_arg[tr.name](tr.arg[0], tr.target)
            elif tr.name in jp_sc_multi_arg:
                jp_no_control_multi_arg[tr.name](*tr.arg, tr.target)
            else:
                raise f"{tr.name} not found in jump table"
        # controls and no args ==> 10
        elif tr.controls and tr.arg is None:
            try:
                jp_sc_no_arg[tr.name](tr.controls[0], tr.target)
            except KeyError:
                raise f"{tr.name} not found in jump table"
        # controls and args ==> 11
        elif tr.controls and tr.arg:
            if tr.name in jp_sc_single_arg:
                jp_sc_single_arg[tr.name](tr.arg[0], tr.controls[0], tr.target)
            elif tr.name in jp_sc_multi_arg:
                jp_sc_multi_arg[tr.name](*tr.arg, tr.controls, tr.target)
            else:
                raise f"{tr.name} not found in jump table"
        else:
            raise f"controls: {tr.controls} and args: {tr.args} did not satisfy any conditions"

    return qc

    # t = 0
    # reg = 0
    # while t >= register_sizes[reg]:
    #     t = t - register_sizes[reg]
    #     reg = reg + 1


def main() -> None:
    """Entry point..."""
    qc_spynoza = simple_spynoza(2)
    transformations = qc_spynoza.get_transformations()
    reg_sizes = qc_spynoza.register_sizes

    qc = spynoza_to_qiskit(reg_sizes, transformations)
    print(qc.draw(output="text"))

    qc_spynoza = all_gates_spynoza()
    transformations = qc_spynoza.get_transformations()
    reg_sizes = qc_spynoza.register_sizes
    qc = spynoza_to_qiskit(reg_sizes, transformations)
    print(qc.draw(output="text"))


if __name__ == "__main__":
    main()
