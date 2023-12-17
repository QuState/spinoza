from spynoza import QuantumCircuit, QuantumRegister, qubit_expectation_value, xyz_expectation_value


def main():
    n = 3
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    target = 0

    qc.rx(0.54, target)
    qc.ry(0.12, target)
    qc.execute()

    targets = list((range(n)))
    exp_vals = xyz_expectation_value('z', qc.get_statevector(), targets)
    print(f"expectation values: {exp_vals}")

    exp_vals = [None] * 3
    for i, target in enumerate(targets):
        exp_vals[i] = qubit_expectation_value(qc.get_statevector(), target)

    print(f"expectation values using `qubit_expectation_value`: {exp_vals}")


if __name__ == "__main__":
    main()