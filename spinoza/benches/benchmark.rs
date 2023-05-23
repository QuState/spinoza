use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use spinoza::{
    core::{iqft, State},
    gates::{apply, c_apply, Gate},
    math::{pow2f, Float, PI},
};

pub fn qcbm_functional(n: usize, pairs: &[(usize, usize)], rng: &mut ThreadRng) {
    let mut state = State::new(n);

    for i in 0..n {
        apply(Gate::RX(rng.gen()), &mut state, i);
        apply(Gate::RZ(rng.gen()), &mut state, i);
    }

    for i in 0..n {
        let (p0, p1) = pairs[i];
        c_apply(Gate::X, &mut state, p0, p1);
    }

    for _ in 0..9 {
        for i in 0..n {
            apply(Gate::RZ(rng.gen()), &mut state, i);
            apply(Gate::RX(rng.gen()), &mut state, i);
            apply(Gate::RZ(rng.gen()), &mut state, i);
        }

        for i in 0..n {
            let (p0, p1) = pairs[i];
            c_apply(Gate::X, &mut state, p0, p1);
        }
    }

    for i in 0..n {
        apply(Gate::RZ(rng.gen()), &mut state, i);
        apply(Gate::RX(rng.gen()), &mut state, i);
    }
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

fn rz_gate(n: usize) {
    let mut state = State::new(n);

    for i in 0..n {
        apply(Gate::RZ(1.0), &mut state, i);
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

fn cx_gate(n: usize, pairs: &[(usize, usize)]) {
    let mut state = State::new(n);

    for i in 0..n {
        let (p0, p1) = pairs[i];
        c_apply(Gate::X, &mut state, p0, p1);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let n = 25;

    c.bench_function("h", |b| b.iter(|| h_gate(black_box(n))));

    c.bench_function("x", |b| b.iter(|| x_gate(black_box(n))));

    let pairs: Vec<_> = (0..n).into_iter().map(|i| (i, (i + 1) % n)).collect();
    c.bench_function("cx", |b| {
        b.iter(|| cx_gate(black_box(n), black_box(&pairs)))
    });

    c.bench_function("rz", |b| b.iter(|| rz_gate(black_box(n))));

    c.bench_function("rx", |b| b.iter(|| rx_gate(black_box(n))));

    let mut rng = rand::thread_rng();
    c.bench_function("qcbm", |b| {
        b.iter(|| qcbm_functional(black_box(n), black_box(&pairs), black_box(&mut rng)))
    });

    c.bench_function("p", |b| b.iter(|| p_gate(black_box(n))));

    c.bench_function("z", |b| b.iter(|| z_gate(black_box(n))));

    c.bench_function("value_encoding", |b| {
        b.iter(|| value_encoding(black_box(n), black_box(2.4)))
    });
}

criterion_group! {name = benches; config = Criterion::default().sample_size(10); targets = criterion_benchmark}
criterion_main!(benches);
