extern crate spinoza;

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use spinoza::{
    circuit::{
        Controls, QuantumCircuit as QuantumCircuitRS, QuantumRegister as QuantumRegisterRS,
        QuantumTransformation as QuantumTransformationRS,
    },
    core::{
        qubit_expectation_value as qubit_expval, reservoir_sampling,
        xyz_expectation_value as xyz_expval, State,
    },
    gates::Gate,
    math::{Amplitude, Float},
    utils::to_table,
};

#[pyclass]
struct PyAmplitude {
    amplitude: Amplitude,
}

#[pymethods]
impl PyAmplitude {
    #[inline]
    pub fn get_complex(&self) -> (Float, Float) {
        (self.amplitude.re, self.amplitude.im)
    }
}

#[pyclass]
pub struct PyState {
    pub data: State,
}

#[pymethods]
impl PyState {
    fn __str__(&self) -> String {
        let mut s = String::with_capacity(self.data.imags.len());
        for (a, b) in self.data.reals.iter().zip(self.data.imags.iter()) {
            s.push_str(&format!("{} + i{}\n", a, b));
        }
        s
    }

    #[inline]
    fn __len__(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn __getitem__(&self, i: usize) -> PyResult<(Float, Float)> {
        Ok((self.data.reals[i], self.data.imags[i]))
    }
}

#[pyfunction]
pub fn show_table(s: &PyState) -> String {
    to_table(&s.data)
}

#[pyclass]
#[derive(Clone)]
pub struct QuantumRegister {
    qr: QuantumRegisterRS,
}

#[pymethods]
impl QuantumRegister {
    #[new]
    pub fn new(size: usize) -> Self {
        Self {
            qr: QuantumRegisterRS((0..size).collect()),
        }
    }

    #[inline]
    fn __len__(&self) -> usize {
        self.qr.len()
    }

    #[inline]
    fn update_shift(&mut self, shift: usize) {
        self.qr.update_shift(shift)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyQuantumTransformation {
    #[pyo3(get, set)]
    pub target: usize,
    #[pyo3(get, set)]
    pub controls: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub arg: Option<(Float, Float, Float)>,
}

#[pymethods]
impl PyQuantumTransformation {
    fn __str__(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("name: {}\n", self.name));
        s.push_str(&format!("target: {}\n", self.target));

        if let Some(a) = self.arg {
            s.push_str(&format!("arg: ({}, {}, {})\n", a.0, a.1, a.2));
        } else {
            s.push_str("arg: None\n");
        }
        if let Some(c) = &self.controls {
            let mut controls_str = String::with_capacity(c.len() + 2);
            controls_str.push('[');
            c.iter().enumerate().for_each(|(i, n)| {
                controls_str.push_str(&n.to_string());
                if i < c.len() - 1 {
                    controls_str.push(',');
                }
            });
            controls_str.push(']');
            s.push_str(&format!("controls: {}", controls_str));
        } else {
            s.push_str("controls: None");
        }
        s
    }
}

impl From<QuantumTransformationRS> for PyQuantumTransformation {
    fn from(item: QuantumTransformationRS) -> Self {
        let target = item.target;

        let controls = match item.controls {
            Controls::Single(q) => Some(vec![q]),
            Controls::Ones(ones) => Some(ones),
            Controls::None => None,
            Controls::Mixed { .. } => todo!(),
        };

        let (name, arg) = match item.gate {
            Gate::H => ("h", None),
            Gate::X => ("x", None),
            Gate::Y => ("y", None),
            Gate::Z => ("z", None),
            Gate::P(angle) => ("p", Some((angle, 0.0, 0.0))),
            Gate::RX(angle) => ("rx", Some((angle, 0.0, 0.0))),
            Gate::RY(angle) => ("ry", Some((angle, 0.0, 0.0))),
            Gate::RZ(angle) => ("rz", Some((angle, 0.0, 0.0))),
            Gate::U(a, b, c) => ("u", Some((a, b, c))),
            _ => todo!(),
        };

        Self {
            target,
            name: name.into(),
            arg,
            controls,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct QuantumTransformation {
    qt: QuantumTransformationRS,
}

#[pyclass]
pub struct QuantumCircuit {
    qc: QuantumCircuitRS,
}

#[pymethods]
impl QuantumCircuit {
    #[new]
    #[pyo3(signature = (*registers))]
    pub fn new(registers: &PyTuple) -> Self {
        let mut regs: Vec<QuantumRegisterRS> = registers
            .into_iter()
            .map(|r| r.extract::<QuantumRegister>().unwrap().qr)
            .collect();

        let mut regs_ref: Vec<&mut QuantumRegisterRS> = regs.iter_mut().collect();
        let regs_ref_slice: &mut [&mut QuantumRegisterRS] = &mut regs_ref;
        Self {
            qc: QuantumCircuitRS::new(regs_ref_slice),
        }
    }

    #[getter(num_qubits)]
    fn num_qubits(&self) -> u8 {
        self.qc.state.n
    }

    #[getter(register_sizes)]
    pub fn register_sizes(&self) -> Vec<usize> {
        self.qc.quantum_registers_info.clone()
    }

    #[getter(state_vector)]
    pub fn state_vector(&self) -> PyResult<PyState> {
        let temp = PyState {
            data: self.qc.state.clone(),
        };
        Ok(temp)
    }

    #[getter(transformations)]
    pub fn transformations(&self) -> Vec<PyQuantumTransformation> {
        self.qc
            .transformations
            .iter()
            .map(|t| t.clone().into())
            .collect()
    }

    #[inline]
    pub fn inverse(&mut self) {
        self.qc.inverse()
    }

    #[inline]
    pub fn h(&mut self, target: usize) {
        self.qc.h(target)
    }

    #[inline]
    pub fn x(&mut self, target: usize) {
        self.qc.x(target)
    }

    #[inline]
    pub fn y(&mut self, target: usize) {
        self.qc.y(target)
    }

    #[inline]
    pub fn z(&mut self, target: usize) {
        self.qc.z(target)
    }

    #[inline]
    pub fn p(&mut self, angle: Float, target: usize) {
        self.qc.p(angle, target)
    }

    #[inline]
    pub fn rx(&mut self, angle: Float, target: usize) {
        self.qc.rx(angle, target)
    }

    #[inline]
    pub fn ry(&mut self, angle: Float, target: usize) {
        self.qc.ry(angle, target)
    }

    #[inline]
    pub fn rz(&mut self, angle: Float, target: usize) {
        self.qc.rz(angle, target)
    }

    #[inline]
    pub fn u(&mut self, theta: Float, phi: Float, lambda: Float, target: usize) {
        self.qc.u(theta, phi, lambda, target)
    }

    // Controlled gates
    pub fn ch(&mut self, control: usize, target: usize) {
        self.qc.ch(control, target)
    }

    #[inline]
    pub fn cx(&mut self, control: usize, target: usize) {
        self.qc.cx(control, target)
    }

    #[inline]
    pub fn ccx(&mut self, control0: usize, control1: usize, target: usize) {
        self.qc.ccx(control0, control1, target)
    }

    #[inline]
    pub fn cy(&mut self, control: usize, target: usize) {
        self.qc.cy(control, target)
    }

    #[inline]
    pub fn cp(&mut self, angle: Float, control: usize, target: usize) {
        self.qc.cp(angle, control, target)
    }

    #[inline]
    pub fn crx(&mut self, angle: Float, control: usize, target: usize) {
        self.qc.crx(angle, control, target)
    }

    #[inline]
    pub fn cry(&mut self, angle: Float, control: usize, target: usize) {
        self.qc.cry(angle, control, target)
    }

    // #[inline]
    pub fn crz(&mut self, angle: Float, control: usize, target: usize) {
        self.qc.crz(angle, control, target)
    }

    // Special gates
    #[inline]
    pub fn measure(&mut self, target: usize) {
        self.qc.measure(target)
    }

    #[inline]
    pub fn swap(&mut self, t0: usize, t1: usize) {
        self.qc.swap(t0, t1)
    }

    #[inline]
    pub fn iqft(&mut self, targets: Vec<usize>) {
        self.qc.iqft(&targets)
    }

    #[inline]
    pub fn add(&mut self, q_transformation: QuantumTransformation) {
        self.qc.add(QuantumTransformationRS {
            gate: q_transformation.qt.gate,
            target: q_transformation.qt.target,
            controls: q_transformation.qt.controls,
        })
    }

    #[inline]
    pub fn append(&mut self, circuit: &QuantumCircuit, quantum_register: &QuantumRegister) {
        self.qc.append(&circuit.qc, &quantum_register.qr)
    }

    #[inline]
    pub fn c_append(
        &mut self,
        circuit: &QuantumCircuit,
        c: usize,
        quantum_register: &QuantumRegister,
    ) {
        self.qc.c_append(&circuit.qc, c, &quantum_register.qr)
    }

    #[inline]
    pub fn mc_append(
        &mut self,
        circuit: &QuantumCircuit,
        cs: Vec<usize>,
        quantum_register: &QuantumRegister,
    ) {
        self.qc.mc_append(&circuit.qc, &cs, &quantum_register.qr)
    }

    #[inline]
    pub fn execute(&mut self) {
        self.qc.execute();
    }
}

// #[pyfunction]
// pub fn combined_probability(state: &PyState, qubits: Vec<usize>) -> Float {
//     comb_prob(&state.data, &qubits)
// }

#[pyfunction]
pub fn xyz_expectation_value(observable: char, state: &PyState, targets: Vec<usize>) -> Vec<Float> {
    xyz_expval(observable, &state.data, &targets)
}

#[pyfunction]
pub fn qubit_expectation_value(state: &PyState, target: usize) -> Float {
    qubit_expval(&state.data, target)
}

#[pyfunction]
pub fn get_samples(
    state: &PyState,
    reservoir_size: usize,
    num_tests: usize,
) -> HashMap<usize, usize> {
    reservoir_sampling(&state.data, reservoir_size, num_tests).get_outcome_count()
}

#[pyfunction]
pub fn run(qc: &mut QuantumCircuit) -> PyResult<PyState> {
    qc.execute();
    qc.state_vector()
}

#[pymodule]
fn spynoza(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(get_samples))?;
    m.add_wrapped(wrap_pyfunction!(show_table))?;
    m.add_wrapped(wrap_pyfunction!(run))?;
    // m.add_wrapped(wrap_pyfunction!(combined_probability))?;
    m.add_wrapped(wrap_pyfunction!(qubit_expectation_value))?;
    m.add_wrapped(wrap_pyfunction!(xyz_expectation_value))?;
    m.add_class::<PyState>()?;
    m.add_class::<QuantumRegister>()?;
    m.add_class::<QuantumCircuit>()?;
    m.add_class::<PyQuantumTransformation>()?;
    Ok(())
}
