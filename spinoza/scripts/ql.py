import numpy as np
import time
from qulacs import QuantumState
from qulacs.gate import to_matrix_gate, H, U1

pi = np.pi
v = 2.4

n = 3

start = time.time()

state = QuantumState(n)
state.set_zero_state()

for i in range(n):
    to_matrix_gate(H(i)).update_quantum_state(state)

for i in range(n):
    if v != 0:
        to_matrix_gate(U1(i, 2 * pi / 2 ** (i + 1) * v)).update_quantum_state(state)

q = range(n)[::-1]
for j in range(n)[::-1]:
    to_matrix_gate(H(q[j])).update_quantum_state(state)
    for k in range(j)[::-1]:
        p_mat_gate = to_matrix_gate(U1(q[k], -np.pi / 2 ** (j - k)))
        p_mat_gate.add_control_qubit(q[j], 1)
        p_mat_gate.update_quantum_state(state)

end = time.time()
print("elapsed time", end - start)

print(state.get_vector())
