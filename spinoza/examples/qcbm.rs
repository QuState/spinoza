use clap::Parser;
use spinoza::{
    config::{Config, QSArgs},
    core::{State, CONFIG},
    gates::{apply, c_apply, Gate},
    utils::{pretty_print_int, to_table},
};

pub fn qcbm_functional(n: usize, show_results: bool) {
    let pairs: Vec<_> = (0..n).into_iter().map(|i| (i, (i + 1) % n)).collect();

    let now = std::time::Instant::now();
    let mut state: State = State::new(n);

    for i in 0..n {
        apply(Gate::RX(1.0), &mut state, i);
        apply(Gate::RZ(1.0), &mut state, i);
    }

    for i in 0..n {
        let (p0, p1) = pairs[i];
        c_apply(Gate::X, &mut state, p0, p1);
    }

    for _ in 0..9 {
        for i in 0..n {
            apply(Gate::RZ(1.0), &mut state, i);
            apply(Gate::RX(1.0), &mut state, i);
            apply(Gate::RZ(1.0), &mut state, i);
        }

        for i in 0..n {
            let (p0, p1) = pairs[i];
            c_apply(Gate::X, &mut state, p0, p1);
        }
    }

    for i in 0..n {
        apply(Gate::RZ(1.0), &mut state, i);
        apply(Gate::RX(1.0), &mut state, i);
    }

    let elapsed = now.elapsed().as_micros();
    println!("{}", pretty_print_int(elapsed));
    if show_results {
        to_table(&state);
    }
}

pub fn main() {
    let args = QSArgs::parse();
    let config = Config::from_cli(args);
    CONFIG.set(config).unwrap();
    qcbm_functional(config.qubits.into(), config.print);
}
