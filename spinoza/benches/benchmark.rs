use criterion::{black_box, criterion_group, criterion_main, Criterion};
use spinoza::{
    core::{iqft, State},
    gates::Gate,
    math::{pow2f, Float, PI},
};

pub fn qcbm_functional(n: usize) {
    let pairs: Vec<_> = (0..n).into_iter().map(|i| (i, (i + 1) % n)).collect();
    let mut state = State::new(n);

    for i in 0..n {
        Gate::RX(1.0).apply(&mut state, i);
        Gate::RZ(1.0).apply(&mut state, i);
    }

    for i in 0..n - 1 {
        let (p0, p1) = pairs[i];
        Gate::X.c_apply(&mut state, p0, p1);
    }

    for _ in 0..9 {
        for i in 0..n {
            Gate::RZ(1.0).apply(&mut state, i);
            Gate::RX(1.0).apply(&mut state, i);
            Gate::RZ(1.0).apply(&mut state, i);
        }

        for i in 0..n - 1 {
            let (p0, p1) = pairs[i];
            Gate::X.c_apply(&mut state, p0, p1);
        }
    }

    for i in 0..n {
        Gate::RZ(1.0).apply(&mut state, i);
        Gate::RX(1.0).apply(&mut state, i);
    }
}

fn value_encoding(n: usize, v: Float) {
    let mut state = State::new(n);

    for i in 0..n {
        Gate::H.apply(&mut state, i);
    }
    for i in 0..n {
        Gate::P(2.0 * PI / (pow2f(i + 1)) * v).apply(&mut state, i);
    }

    let targets: Vec<usize> = (0..n).rev().collect();
    iqft(&mut state, &targets);
}

fn h_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        Gate::H.apply(&mut state, i);
    }
}

fn rx_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        Gate::RX(1.0).apply(&mut state, i);
    }
}

fn rz_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        Gate::RZ(1.0).apply(&mut state, i);
    }
}

fn x_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        Gate::X.apply(&mut state, i);
    }
}

fn p_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        Gate::P(1.0).apply(&mut state, i);
    }
}

fn z_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        Gate::Z.apply(&mut state, i);
    }
}

fn cx_gate(n: usize) {
    let pairs: Vec<_> = (0..n).into_iter().map(|i| (i, (i + 1) % n)).collect();
    let mut state = State::new(n);

    for i in 0..n {
        let (p0, p1) = pairs[i];
        Gate::X.c_apply(&mut state, p0, p1);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("h", |b| b.iter(|| h_gate(black_box(25))));

    c.bench_function("x", |b| b.iter(|| x_gate(black_box(25))));

    c.bench_function("cx", |b| b.iter(|| cx_gate(black_box(25))));

    c.bench_function("rz", |b| b.iter(|| rz_gate(black_box(25))));

    c.bench_function("rx", |b| b.iter(|| rx_gate(black_box(25))));

    c.bench_function("qcbm", |b| b.iter(|| qcbm_functional(black_box(25))));

    c.bench_function("p", |b| b.iter(|| p_gate(black_box(25))));

    c.bench_function("z", |b| b.iter(|| z_gate(black_box(25))));

    c.bench_function("value_encoding", |b| {
        b.iter(|| value_encoding(black_box(20), black_box(2.4)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
