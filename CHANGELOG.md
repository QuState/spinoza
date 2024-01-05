# CHANGELOG

## [0.5.1] - 2024-01-04

### Performance Improvements

- **Measurement**:
    - Improved performance of measurement by almost 50% in certain cases. Rayon will no longer
    splitting the iterator when the number of qubits is low (i.e., when # of qubits < 16).
    - Improved performance by removing superfluous instructions. Replaced
    modulus and squaring with direct computation for efficiency.

### Code Refactoring

- Streamlined the implementation of measure_qubit by utilizing the `take` iterator.

### Fixed

- Fixed spynoza examples to align with API changes introduced in v0.5.0.
- Resolved an issue with a configuration test.

### Other Changes

- Update clap to 4.4.13
- Update pyo3 to 0.20.2

## [0.5.0] - 2023-12-31

### Breaking Changes

- **Updated Gate Enum:**
  - The `Gate` enum has undergone breaking changes for improved usability and clarity:
    - Removed tuple parameters from `SWAP` gate; it now takes two separate `usize` parameters.
    - Removed tuple parameters from `U` gate; it now takes three separate `Float` parameters.
    - Added a new gate: `BitFlipNoise(Float)` for simulating a bit flip based on the provided probability.

### New Features

- **Controlled Hadamard Gate:**
  - Addition of the controlled Hadamard gate for enhanced quantum circuit design.

- **Controlled Rz Gate:**
  - Addition of the controlled Rz gate to broaden the range of available quantum gates.

- **Controlled U Gate:**
  - Addition of the controlled U gate, allowing controlled application of a general single-qubit rotation.

- **Expectation Value Functions:**
  - **`xyz_expectation_value`:**
    - New function allowing computation of expectation value for any of the operators `X, Y, Z`.
  - **`qubit_expectation_value`:**
    - Specialized function, for improved performance, to compute the expectation value of `Z` in the state.

### Other Changes

- Improved Documentation:
  - Enhanced documentation for new features and functions to assist users in integrating them into their projects.

### Deprecated

- None

### Removed

- None

### Fixed

- None

### Notes for Developers

- Developers should be aware that breaking changes have been introduced to the `Gate` enum in this release. Please 
  review and update your code accordingly. Breaking changes will not result in a major version increment until `v1.0.0`.

## 0.4.0

- The default number of threads is now determined using
  std::thread::available_parallelism in lieu of using num_cpus.
- `once_cell` has become a part of the standard library since Rust
  1.70.0 so usage of `OnceCell` has now been replaced with `OnceLock`.

## 0.3.0

- Add example for sampling
- Parallelize sampling and add sampling tests
- `to_table()` now returns a `String` in lieu of just printing the table
- Provide improved examples for python interface users
- Update dependencies to latest available releases
- Update python bindings to reflect latest changes/features
- Add code coverage to CI pipeline and add codecov badge to README
- Add instructions for building on macOS
- Add support for classical control
- Add support to convert a subset of OpenQASM 2.0 programs into `QuantumCircuit`
- Add inverse of `QuantumCircuit` via the `inverse()` method
- Improve performance of measurement using iterators and rayon
- Add support for `Gate` inverses
- Improve performance of the SWAP gate
- Add support for applying Unitaries to the state vector
- Add single qubit measurement
- Fix bug in U gate @nfwvogt
- Add support for parallel execution
- Move benchmarks to separate repository
