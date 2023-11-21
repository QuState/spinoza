use clap::Parser;
use spinoza::{
    circuit::{QuantumCircuit, QuantumRegister},
    config::{Config, QSArgs},
    core::CONFIG,
    math::{pow2f, PI},
    utils::{pretty_print_int, to_table},
};

fn benchmark_circuit_value_encoding(n: usize, show_results: bool) {
    let v = 2.4;
    let now = std::time::Instant::now();
    let mut q = QuantumRegister::new(n);
    let mut qc = QuantumCircuit::new(&mut [&mut q]);

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
        println!("{}", to_table(qc.get_statevector()));
    }
}

pub fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();
    benchmark_circuit_value_encoding(config.qubits.into(), config.print);
}
