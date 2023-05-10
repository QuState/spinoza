#define _USE_MATH_DEFINES
#include <math.h>  // cos() sin()
#include <omp.h>
#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string.h>
#include <tuple>
#include <vector>
#include "utils.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

struct Amplitude {
    double re;
    double im;
};

struct State {
    std::vector<Amplitude> data;
};

State init(int n) {
    if (n < 1) {
        std::cerr << "state must be greater than 0" << std::endl;
    }

    std::vector<Amplitude> data(1 << n);

#pragma omp parallel for
    for (int i = 1; i < (1 << n); i++) {
        data[i] = Amplitude{ 0.0, 0.0 };
    }

    data[0] = Amplitude{ 1.0, 0.0 };

    return State{data};
}

struct RX {
    Amplitude matrix[4];
};

RX new_rx(double angle) {
    double theta = angle * 0.5;
    const auto c = Amplitude{cos(theta), 0.0};
    const auto ns = Amplitude{0.0, -sin(theta)};

    RX rx;
    rx.matrix[0] = c;
    rx.matrix[1] = ns;
    rx.matrix[2] = ns;
    rx.matrix[3] = c;

    return rx;
}

inline void _apply_target_0(const RX rx, Amplitude* __restrict__ data, const size_t basis) {
    const auto x = data[basis].re;
    const auto y = data[basis].im;

    const auto m = data[basis + 1].re;
    const auto n = data[basis + 1].im;

    data[basis].re = rx.matrix[0].re * x - rx.matrix[1].im * n;
    data[basis].im = rx.matrix[0].re * y + rx.matrix[1].im * m;

    data[basis + 1].re = rx.matrix[0].re * x - rx.matrix[1].im * n;
    data[basis + 1].im = rx.matrix[0].re * y + rx.matrix[1].im * m;
}

inline void _apply_target(const RX rx, Amplitude* __restrict__ data, const size_t basis_0, const size_t basis_1) {
    const auto x0 = data[basis_0].re;
    const auto y0 = data[basis_0].im;

    const auto m0 = data[basis_1].re;
    const auto n0 = data[basis_1].im;

    const auto x1 = data[basis_0 + 1].re;
    const auto y1 = data[basis_0 + 1].im;

    const auto m1 = data[basis_1 + 1].re;
    const auto n1 = data[basis_1 + 1].im;

    data[basis_0].re = rx.matrix[0].re * x0 - rx.matrix[1].im * n0;
    data[basis_0].im = rx.matrix[0].re * y0 + rx.matrix[1].im * m0;

    data[basis_1].re = (-rx.matrix[2].im) * y0 + rx.matrix[3].re * m0;
    data[basis_1].im = rx.matrix[2].im * x0 + rx.matrix[3].re * n0;

    data[basis_0 + 1].re = rx.matrix[0].re * x1 - rx.matrix[1].im * n1;
    data[basis_0 + 1].im = rx.matrix[0].re * y0 + rx.matrix[1].im * m1;

    data[basis_1 + 1].re = (-rx.matrix[2].im) * y1 + rx.matrix[3].re * m1;
    data[basis_1 + 1].im = rx.matrix[2].im * x1 + rx.matrix[3].re * n1;
}

void apply(const RX rx, State& state, size_t target_qubit_index) {
    const auto mask = 1 << target_qubit_index;
    const auto mask_low = mask - 1;
    const auto mask_high = ~mask_low;

    if (target_qubit_index == 0) {
#pragma omp parallel for
        for (int basis = 0; basis < state.data.size(); basis += 2) {
            _apply_target_0(rx, state.data.data(), basis);
        }
    } else {
        const auto loop_dim = state.data.size() / 2;
#pragma omp parallel for
        for (int state_index = 0; state_index < loop_dim; state_index += 2) {
            const auto basis_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            const auto basis_1 = basis_0 + mask;
            _apply_target(rx, state.data.data(), basis_0, basis_1);
        }
    }
}


void benchmark1(size_t n) {
    auto t1 = high_resolution_clock::now();
    State state = init(n);
    auto t2 = high_resolution_clock::now();
    auto us_int = duration_cast<microseconds>(t2 - t1);
    std::cout << "state initialized in " << FormatWithCommas(us_int.count()) << " us" << "\n";

    const auto rx_gate = new_rx(1.0);

    for (int t = 0; t < n; t++) {
        auto t1 = high_resolution_clock::now();
        apply(rx_gate, state, t);
        auto t2 = high_resolution_clock::now();
        auto us_int = duration_cast<microseconds>(t2 - t1);
        std::cout << "applied target " << t << " in " << FormatWithCommas(us_int.count()) << " us" << "\n";
    }

}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <num-qubits>" << std::endl;
        return EXIT_FAILURE;
    }

    int n = std::stoi(argv[1]);
    benchmark1(n);

    return EXIT_SUCCESS;
}
