[![Build](https://github.com/QuState/spinoza/actions/workflows/rust.yml/badge.svg)](https://github.com/QuState/spinoza/actions/workflows/rust.yml)
[![Spinoza documentation](https://img.shields.io/badge/docs-passing-brightgreen)](https://qustate.github.io/doc/spinoza/)

<img src="assets/logo.png" width="500">

## Getting Started

### Prerequisites

[Rust](https://www.rust-lang.org/learn/get-started)

[nightly Rust](https://rust-lang.github.io/rustup/concepts/channels.html)
```bash
rustup toolchain install nightly
rustup default nightly
```

### Building on *nix

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
