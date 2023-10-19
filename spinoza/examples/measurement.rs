use clap::Parser;
use spinoza::{
    config::{Config, QSArgs},
    core::{State, CONFIG},
    gates::{apply, Gate},
    measurement::measure_qubit,
    utils::{pretty_print_int, to_table},
};

fn h(n: usize, show_results: bool) {
    let now = std::time::Instant::now();
    let mut state = State::new(n);

    for t in 0..n {
        apply(Gate::H, &mut state, t);
    }

    let elapsed = now.elapsed().as_micros();
    println!("{}", pretty_print_int(elapsed));

    let now = std::time::Instant::now();
    measure_qubit(&mut state, n - 1, true, Some(1));
    let elapsed = now.elapsed().as_micros();
    println!("{}", pretty_print_int(elapsed));
}

fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();
    h(config.qubits.into(), config.print);
}
