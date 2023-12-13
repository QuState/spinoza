# CHANGELOG

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
