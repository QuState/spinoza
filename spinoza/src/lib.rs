#![feature(avx512_target_feature)]
//! A High Performance Quantum State Simulator
//!
//! Spinoza is a fast and flexible quantum simulator written exclusively in Rust, with bindings available for Python
//! users. Spinoza simulates the evolution of a quantum system’s state by applying quantum gates, with the core design
//! principle being that a single-qubit gate applied to a target qubit preserves the probability of pairs of amplitudes
//! corresponding to measurement outcomes that differ only in the target qubit. Spinoza is intended to enable the
//! development of quantum computing solutions by offering researchers and quantum developers a simple, flexible, and
//! fast tool for classical simulation.
//!
//! # How to use Spinoza
//!
//! There are three ways to use Spinoza:
//!
//! - **Functional** the simplest way to mutate a quantum state, directly.
//!   - [apply][gates::apply] for quantum transformations that do not need a control (qubit).
//!   - [c_apply][gates::c_apply] for quantum transformations that have a single control.
//!   - [mc_apply][gates::mc_apply] for quantum transformation that require multiple controls.
//! - **Object-Oriented** the [QuantumCircuit][circuit::QuantumCircuit] represents a [quantum circuit](https://en.wikipedia.org/wiki/Quantum_circuit).
//!   Using the [QuantumCircuit][circuit::QuantumCircuit] one can create, mutate, and simulate quantum circuits with
//!   various quantum [gates][gates::Gate], operators, etc.
//! - **Python Bindings** Spinoza has python bindings named `Spynoza`
//!   - All functionality for [QuantumCircuit][circuit::QuantumCircuit] and other functions have corresponding bindings
//!     created using PyO3.
//!
//! # Examples
//! Let's encode the value, 2.4 using the three aforementioned approaches:
//!
//! ### Functional
//! ```
//! use spinoza::{
//!     core::{iqft, State},
//!     gates::{apply, Gate},
//!     math::{pow2f, PI},
//!     utils::{to_table},
//! };
//!
//! pub fn main() {
//!     let n = 3;
//!     let v = 2.4;
//!     let mut state = State::new(n);
//!
//!     for i in 0..n {
//!         apply(Gate::H, &mut state, i);
//!     }
//!    for i in 0..n {
//!         apply(Gate::P(2.0 * PI / (pow2f(i + 1)) * v), &mut state, i);
//!     }
//!     let targets: Vec<usize> = (0..n).rev().collect();
//!
//!     iqft(&mut state, &targets);
//!     println!("{}", to_table(&state));
//! }
//! ```
//! ### Object Oriented (OO)
//! ```
//! use spinoza::{
//!     core::{iqft, State},
//!     circuit::{QuantumCircuit, QuantumRegister},
//!     math::{pow2f, PI},
//!     utils::{to_table},
//! };
//!
//! pub fn main() {
//!     let n = 3;
//!     let v = 2.4;
//!     let now = std::time::Instant::now();
//!     let mut q = QuantumRegister::new(n);
//!     let mut qc = QuantumCircuit::new(&mut [&mut q]);
//!
//!     for i in 0..n {
//!         qc.h(i)
//!     }
//!     for i in 0..n {
//!         qc.p(2.0 * PI / pow2f(i + 1) * v, i)
//!     }
//!
//!     let targets: Vec<usize> = (0..n).rev().collect();
//!     qc.iqft(&targets);
//!     qc.execute();
//!     println!("{}", to_table(qc.get_statevector()));
//! }
//!```
//! ### Spynoza
//! ```python
//! from math import pi
//! from spynoza import QuantumCircuit, QuantumRegister, show_table
//!
//!
//! def value_encoding(n, v):
//!     q = QuantumRegister(n)
//!     qc = QuantumCircuit(q)
//!
//!     for i in range(n):
//!         qc.h(i)
//!
//!     for i in range(n):
//!         qc.p(2 * pi / (2 ** (i + 1)) * v, i)
//!
//!     qc.iqft(range(n)[::-1])
//!
//!     qc.execute()
//!     return qc.get_statevector()
//!
//!
//! if __name__ == "__main__":
//!     state = value_encoding(4, 2.4)
//!     print(show_table(state))
//!```
//!
//! More complex examples can be found in the [Spinoza examples](https://github.com/QuState/spinoza/tree/main/spinoza/examples)
//! and the [Spynoza exmaples](https://github.com/QuState/spinoza/tree/main/spynoza/examples).
//!
//! # References
//! ```latex
//! @misc{yusufov2023designing,
//!       title={Designing a Fast and Flexible Quantum State Simulator},
//!       author={Saveliy Yusufov and Charlee Stefanski and Constantin Gonciulea},
//!       year={2023},
//!       eprint={2303.01493},
//!       archivePrefix={arXiv},
//!       primaryClass={quant-ph}
//! }
//!```

#![warn(clippy::complexity)]
#![warn(missing_docs)]
#![warn(clippy::style)]
#![warn(clippy::correctness)]
#![warn(clippy::suspicious)]
#![warn(clippy::perf)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod circuit;
pub mod config;
pub mod consts;
pub mod core;
pub mod gates;
pub mod math;
pub mod measurement;
pub mod openqasm;
pub mod unitaries;
pub mod utils;
