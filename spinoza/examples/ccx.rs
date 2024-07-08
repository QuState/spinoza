use clap::Parser;
use spinoza::{
    config::{Config, QSArgs},
    core::{State, CONFIG},
    gates::{apply, cc_apply, Gate},
    math::PI,
    utils::{pretty_print_int, to_table},
};

fn x(n: usize, show_results: bool) {
    let now = std::time::Instant::now();
    let mut state = State::new(n);

    for i in 0..n {
        apply(Gate::H, &mut state, i);
        apply(Gate::RZ(PI), &mut state, i);
    }

    if show_results {
        to_table(&state);
    }

    cc_apply(Gate::X, &mut state, 0, 2, 1);

    let elapsed = now.elapsed().as_micros();
    println!("{}", pretty_print_int(elapsed));
    if show_results {
        println!("{}", to_table(&state));
    }
}

fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();
    x(config.qubits.into(), config.print);
}
