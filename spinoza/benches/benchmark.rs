use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use spinoza::{
    circuit::{QuantumCircuit, QuantumRegister},
    core::{iqft, State},
    gates::{apply, c_apply, Gate},
    math::{pow2f, Float, PI},
    measurement::measure_qubit,
    utils::{gen_random_state, pretty_print_int},
};

fn first_rotation(circuit: &mut QuantumCircuit, nqubits: usize, angles: &mut Vec<Float>) {
    for k in 0..nqubits {
        circuit.rx(angles.pop().unwrap(), k);
        circuit.rz(angles.pop().unwrap(), k);
    }
}

fn mid_rotation(circuit: &mut QuantumCircuit, nqubits: usize, angles: &mut Vec<Float>) {
    for k in 0..nqubits {
        circuit.rz(angles.pop().unwrap(), k);
        circuit.rx(angles.pop().unwrap(), k);
        circuit.rz(angles.pop().unwrap(), k);
    }
}

fn last_rotation(circuit: &mut QuantumCircuit, nqubits: usize, angles: &mut Vec<Float>) {
    for k in 0..nqubits {
        circuit.rz(angles.pop().unwrap(), k);
        circuit.rx(angles.pop().unwrap(), k);
    }
}

fn entangler(circuit: &mut QuantumCircuit, pairs: &[(usize, usize)]) {
    for (a, b) in pairs.iter() {
        circuit.cx(*a, *b);
    }
}

fn build_circuit(nqubits: usize, depth: usize, pairs: &[(usize, usize)]) -> QuantumCircuit {
    let mut rng = StdRng::seed_from_u64(42);
    let mut angles: Vec<_> = (0..(nqubits * 2) + (depth * nqubits * 3) + (nqubits * 2))
        .map(|_| rng.gen())
        .collect();

    let mut q = QuantumRegister::new(nqubits);
    let mut circuit = QuantumCircuit::new(&mut [&mut q]);
    first_rotation(&mut circuit, nqubits, &mut angles);
    entangler(&mut circuit, pairs);
    for _ in 0..depth {
        mid_rotation(&mut circuit, nqubits, &mut angles);
        entangler(&mut circuit, pairs);
    }

    last_rotation(&mut circuit, nqubits, &mut angles);
    circuit
}

pub fn qcbm(circuit: &mut QuantumCircuit) {
    circuit.execute();
}

fn value_encoding(n: usize, v: Float) {
    let mut state = State::new(n);

    for i in 0..n {
        apply(Gate::H, &mut state, i);
    }
    for i in 0..n {
        apply(Gate::P(2.0 * PI / (pow2f(i + 1)) * v), &mut state, i);
    }

    let targets: Vec<usize> = (0..n).rev().collect();
    iqft(&mut state, &targets);
}

fn h_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        apply(Gate::H, &mut state, i);
    }
}

fn rx_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        apply(Gate::RX(1.0), &mut state, i);
    }
}

fn rz_gate(state: &mut State, n: usize) {
    for i in 0..n {
        apply(Gate::RZ(1.0), state, i);
    }
}

fn x_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        apply(Gate::X, &mut state, i);
    }
}

fn p_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        apply(Gate::P(1.0), &mut state, i);
    }
}

fn z_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        apply(Gate::Z, &mut state, i);
    }
}

fn cx_gate(state: &mut State, n: usize, pairs: &[(usize, usize)]) {
    for i in 0..n {
        let (p0, p1) = pairs[i];
        c_apply(Gate::X, state, p0, p1);
    }
}

fn u_gate(n: usize) {
    let mut state = State::new(n);

    for t in 0..n {
        apply(Gate::U(1.0, 2.0, 3.0), &mut state, t);
    }
}

fn pprint_int(i: u128) {
    let _res = pretty_print_int(i);
}

fn measure(n: usize) {
    let mut state = gen_random_state(n);
    measure_qubit(&mut state, 0, true, None);
}

fn criterion_benchmark(c: &mut Criterion) {
    let n = 25;

    c.bench_function("h", |b| b.iter(|| h_gate(black_box(n))));

    c.bench_function("x", |b| b.iter(|| x_gate(black_box(n))));

    let mut state = State::new(n);
    let pairs: Vec<_> = (0..n).into_iter().map(|i| (i, (i + 1) % n)).collect();
    c.bench_function("cx", |b| {
        b.iter(|| cx_gate(black_box(&mut state), black_box(n), black_box(&pairs)))
    });

    let mut state = State::new(n);
    c.bench_function("rz", |b| {
        b.iter(|| rz_gate(black_box(&mut state), black_box(n)))
    });

    c.bench_function("rx", |b| b.iter(|| rx_gate(black_box(n))));

    let mut circuit = build_circuit(n, 9, &pairs);
    c.bench_function("qcbm", |b| b.iter(|| qcbm(&mut circuit)));

    c.bench_function("p", |b| b.iter(|| p_gate(black_box(n))));

    c.bench_function("z", |b| b.iter(|| z_gate(black_box(n))));

    c.bench_function("u", |b| b.iter(|| u_gate(black_box(n))));

    c.bench_function("value_encoding", |b| {
        b.iter(|| value_encoding(black_box(n), black_box(2.4)))
    });

    c.bench_function("pprint_int", |b| {
        b.iter(|| pprint_int(black_box(u128::MAX)))
    });

    c.bench_function("measure", |b| b.iter(|| measure(black_box(n))));
}

criterion_group! {name = benches; config = Criterion::default().sample_size(100); targets = criterion_benchmark}
criterion_main!(benches);
