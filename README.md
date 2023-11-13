[![Build](https://github.com/QuState/spinoza/actions/workflows/rust.yml/badge.svg)](https://github.com/QuState/spinoza/actions/workflows/rust.yml)
[![Spinoza documentation](https://img.shields.io/badge/docs-passing-brightgreen)](https://qustate.github.io/spinoza/)

<p align="center">
<img src="assets/logo.png" width="400">
</p>

Spinoza is a fast and flexible quantum simulator written exclusively in Rust,
with bindings available for Python users. Spinoza simulates the evolution of a
quantum system’s state by applying quantum gates, with the core design
principle being that a single-qubit gate applied to a target qubit preserves
the probability of pairs of amplitudes corresponding to measurement outcomes
that differ only in the target qubit. Spinoza is intended to enable the
development of quantum computing solutions by offering researchers and quantum
develoers a simple, flexible, and fast tool for classical simulation. For more
information, please see the accompanying
[paper](https://arxiv.org/pdf/2303.01493.pdf).

## Getting Started

### Prerequisites

[Rust](https://www.rust-lang.org/learn/get-started)

[nightly Rust](https://rust-lang.github.io/rustup/concepts/channels.html)
```bash
rustup toolchain install nightly
rustup default nightly
```

### Building on *nix [for macOS]](INSTALL.md)

#### Develoment
```bash
cargo build
```

#### Production
```bash
cargo build --release
```

### Testing
```bash
cargo test
```

### Try it out!

In general, examples can be run using:
```bash
cargo run --release --example <example-name> -- -q <num-qubits>
```

For example:
```bash
cargo run --release --example rx -- --threads $(nproc) --qubits 20
cargo run --release --example ry -- --threads $(nproc) --qubits 20
cargo run --release --example rz -- --threads $(nproc) --qubits 20
cargo run --release --example x -- --threads $(nproc) --qubits 20
cargo run --release --example z -- --threads $(nproc) --qubits 20
cargo run --release --example p -- --threads $(nproc) --qubits 20
```

___

## References
```
@misc{yusufov2023designing,
      title={Designing a Fast and Flexible Quantum State Simulator},
      author={Saveliy Yusufov and Charlee Stefanski and Constantin Gonciulea},
      year={2023},
      eprint={2303.01493},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
