# CHANGELOG

## 0.3.0

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
