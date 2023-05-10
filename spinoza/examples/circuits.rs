use clap::Parser;
use spinoza::{
    circuit::{QuantumCircuit, QuantumRegister},
    config::{Config, QSArgs},
    core::CONFIG,
    math::{pow2f, PI},
    utils::{pretty_print_int, to_table},
};

// fn benchmark_x(n: usize) {
//     let q = &mut QuantumRegister::new(n);
//     let mut qc = QuantumCircuit::new(q);
//
//     qc.x(3);
//
//     let _ = qc.run();
// }
//
// fn benchmark_h(n: usize) {
//     let q = &mut QuantumRegister::new(n);
//     let mut qc = QuantumCircuit::new(q);
//
//     qc.h(3);
//
//     let _ = qc.run();
// }
//
// fn benchmark_t(n: usize) {
//     let q = &mut QuantumRegister::new(n);
//     let mut qc = QuantumCircuit::new(q);
//
//     qc.p(PI / 4.0, 3);
//
//     let _ = qc.run();
// }
//
// fn benchmark_rx(n: usize) {
//     let q = &mut QuantumRegister::new(n);
//     let mut qc = QuantumCircuit::new(q);
//
//     for i in 0..n {
//         qc.rx(PI / 4.0, i);
//     }
//
//     let _ = qc.run();
// }
//
// fn benchmark_rz(n: usize) {
//     let q = &mut QuantumRegister::new(n);
//     let mut qc = QuantumCircuit::new(q);
//
//     qc.rz(0.5, 3);
//
//     let _ = qc.run();
// }
//
// fn benchmark_cnot(n: usize) {
//     let q = &mut QuantumRegister::new(n);
//     let mut qc = QuantumCircuit::new(q);
//
//     qc.cx(1, 2);
//
//     let _ = qc.run();
// }
//
// fn benchmark_toffoli(n: usize) {
//     let q = &mut QuantumRegister::new(n);
//     let mut qc = QuantumCircuit::new(q);
//
//     qc.ccx(2, 3, 0);
//     let _ = qc.run();
// }
//
// pub fn benchmark_qcbm(n: usize) {
//     let now = std::time::Instant::now();
//
//     let pairs: Vec<_> = (0..n).into_iter().map(|i| (i, (i + 1) % n)).collect();
//     let q = &mut QuantumRegister::new(n);
//     let mut qc = QuantumCircuit::new(q);
//
//     for i in 0..n {
//         qc.rx(1.0, i);
//         qc.rz(1.0, i);
//     }
//
//     for i in 0..n - 1 {
//         let (p0, p1) = pairs[i];
//         qc.cx(p0, p1);
//     }
//
//     for _ in 0..9 {
//         for i in 0..n {
//             qc.rz(1.0, i);
//             qc.rx(1.0, i);
//             qc.rz(1.0, i);
//         }
//
//         for i in 0..n - 1 {
//             let (p0, p1) = pairs[i];
//             qc.cx(p0, p1);
//         }
//     }
//
//     for i in 0..n {
//         qc.rz(1.0, i);
//         qc.rx(1.0, i);
//     }
//     let state = qc.run();
//
//     let elapsed = now.elapsed().as_micros();
//     println!("{}", elapsed);
// }

fn benchmark_circuit_value_encoding(n: usize, show_results: bool) {
    let v = 2.4;
    let now = std::time::Instant::now();
    let mut q = QuantumRegister::new(n);
    let mut qc = QuantumCircuit::new(&mut q);

    for i in 0..n {
        qc.h(i)
    }
    for i in 0..n {
        qc.p(2.0 * PI / pow2f(i + 1) * v, i)
    }

    let targets: Vec<usize> = (0..n).rev().collect();
    qc.iqft(&targets);
    qc.execute();

    let elapsed = now.elapsed().as_micros();
    println!("{}", pretty_print_int(elapsed));

    if show_results {
        to_table(qc.get_statevector())
    }
}

pub fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();
    benchmark_circuit_value_encoding(config.qubits.into(), config.print);
}
