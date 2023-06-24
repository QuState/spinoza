use clap::Parser;
use spinoza::{
    config::{Config, QSArgs},
    core::{State, CONFIG},
    gates::{apply, Gate},
    utils::{pretty_print_int, to_table},
};

fn x(n: usize, show_results: bool) {
    let now = std::time::Instant::now();
    let mut state = State::new(n);
    let elapsed = now.elapsed().as_micros();
    println!(
        "created state of {} qubits in {} us",
        n,
        pretty_print_int(elapsed)
    );

    for t in 0..n {
        let now = std::time::Instant::now();
        apply(Gate::X, &mut state, t);
        let elapsed = now.elapsed().as_micros();
        println!(
            "applied X for target {} in {} us",
            t,
            pretty_print_int(elapsed)
        );
    }

    // let elapsed = now.elapsed().as_micros();
    // println!("{}", pretty_print_int(elapsed));
    if show_results {
        to_table(&state);
    }
}

fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();
    x(config.qubits.into(), config.print);
}
