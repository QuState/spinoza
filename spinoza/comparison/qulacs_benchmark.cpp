#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include "utils.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::microseconds;

void benchmark_value_encoding(int n, float v) {
    QuantumState state(n);
    state.set_zero_state();

    for (int i = 0; i < n; i++) {
        gate::to_matrix_gate(gate::H(i))->update_quantum_state(&state);
    }

    for (int i = 0; i < n; i++) {
        if (v != 0) {
            gate::to_matrix_gate(gate::U1(i, 2 * M_PI / (1 << (i + 1)) * v))->update_quantum_state(&state);
        }
    }

    std::vector<int> q;
    for (int i = n-1; i > -1; i--) {
        q.push_back(i);
    }

    for (int j = n-1; j > -1; j--) {
        gate::to_matrix_gate(gate::H(q[j]))->update_quantum_state(&state);

        for (int k = j-1; k > -1; k--) {
            auto p_mat_gate = gate::to_matrix_gate(gate::U1(q[k], -M_PI / (1 << (j - k))));
            p_mat_gate->add_control_qubit(q[j], 1);
            p_mat_gate->update_quantum_state(&state);
        }
    }
}

void benchmark_qcbm(int n) {
    QuantumState state(n);
    state.set_zero_state();
    QuantumCircuit circuit(n);

    std::vector<std::tuple<int, int>> pairs;
    for (int i = 0; i < n; i++) {
        auto p = std::make_tuple(i, (i + 1) % n);
        pairs.push_back(p);
    }

    for (int i = 0; i < n; i++) {
        circuit.add_gate(gate::RX(i, 1.0));
        circuit.add_gate(gate::RZ(i, 1.0));
    }

    for (int i = 0; i < n-1; i++) {
        auto p = pairs[i];
        auto p0 = std::get<0>(p);
        auto p1 = std::get<1>(p);

        auto x_mat_gate = gate::to_matrix_gate(gate::X(p1));
        unsigned int control_index = p0;
        unsigned int control_with_value = 1;
        x_mat_gate->add_control_qubit(control_index, control_with_value);
        circuit.add_gate(x_mat_gate);
    }

    for (int d = 0; d < 9; d++) {
        for (int i = 0; i < n; i++) {
            circuit.add_gate(gate::RZ(i, 1.0));
            circuit.add_gate(gate::RX(i, 1.0));
            circuit.add_gate(gate::RZ(i, 1.0));
        }

        for (int i = 0; i < n-1; i++) {
            auto p = pairs[i];
            auto p0 = std::get<0>(p);
            auto p1 = std::get<1>(p);

            auto x_mat_gate = gate::to_matrix_gate(gate::X(p1));
            unsigned int control_index = p0;
            unsigned int control_with_value = 1;
            x_mat_gate->add_control_qubit(control_index, control_with_value);
            circuit.add_gate(x_mat_gate);
        }
    }

    for (int i = 0; i < n; i++) {
        circuit.add_gate(gate::RZ(i, 1.0));
        circuit.add_gate(gate::RX(i, 1.0));
    }
    circuit.update_quantum_state(&state);
    // std::cout << state << "\n";
}

void benchmark_qcbm_functional(int n) {
    auto t1 = high_resolution_clock::now();

    QuantumState state(n);
    state.set_zero_state();

    std::vector<std::tuple<int, int>> pairs;
    for (int i = 0; i < n; i++) {
        auto p = std::make_tuple(i, (i + 1) % n);
        pairs.push_back(p);
    }

    for (int i = 0; i < n; i++) {
        gate::to_matrix_gate(gate::RX(i, 1.0))->update_quantum_state(&state);
        gate::to_matrix_gate(gate::RZ(i, 1.0))->update_quantum_state(&state);
    }

    for (int i = 0; i < n - 1; i++) {
        auto p = pairs[i];
        auto p0 = std::get<0>(p);
        auto p1 = std::get<1>(p);

        auto x_mat_gate = gate::to_matrix_gate(gate::X(p1));
        x_mat_gate->add_control_qubit(p0, 1);
        x_mat_gate->update_quantum_state(&state);
    }

    for (int d = 0; d < 9; d++) {
        for (int i = 0; i < n; i++) {
            gate::to_matrix_gate(gate::RZ(i, 1.0))->update_quantum_state(&state);
            gate::to_matrix_gate(gate::RX(i, 1.0))->update_quantum_state(&state);
            gate::to_matrix_gate(gate::RZ(i, 1.0))->update_quantum_state(&state);
        }

        for (int i = 0; i < n - 1; i++) {
            auto p = pairs[i];
            auto p0 = std::get<0>(p);
            auto p1 = std::get<1>(p);
            auto x_mat_gate = gate::to_matrix_gate(gate::X(p1));
            x_mat_gate->add_control_qubit(p0, 1);
            x_mat_gate->update_quantum_state(&state);
        }
    }

    for (int i = 0; i < n; i++) {
        gate::to_matrix_gate(gate::RZ(i, 1.0))->update_quantum_state(&state);
        gate::to_matrix_gate(gate::RX(i, 1.0))->update_quantum_state(&state);
    }

    auto t2 = high_resolution_clock::now();
    auto us_int = duration_cast<microseconds>(t2 - t1);
    std::cout << n << ", " << FormatWithCommas(us_int.count()) << "\n";

    auto raw_data_c = state.data_cpp();
    for (int i = 0; i < std::min(16, 1 << n); i++) {
        std::cout << i << ": " << raw_data_c[i] << "\n";
    }
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <num-qubits>" << std::endl;;
        return EXIT_FAILURE;
    }

    int n = std::stoi(argv[1]);
    benchmark_qcbm_functional(n);

    return EXIT_SUCCESS;
}
