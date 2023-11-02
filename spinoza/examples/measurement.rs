use clap::Parser;
use spinoza::{
    config::{Config, QSArgs},
    core::CONFIG,
    measurement::measure_qubit,
    utils::{gen_random_state, pretty_print_int},
};

fn measure_qubits(n: usize) {
    let mut state = gen_random_state(n);

    let now = std::time::Instant::now();
    for t in 0..n {
        measure_qubit(&mut state, t, true, None);
    }
    let elapsed = now.elapsed().as_micros();
    println!("measured all qubits in {} us", pretty_print_int(elapsed));
}

fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();
    measure_qubits(config.qubits.into());
}
