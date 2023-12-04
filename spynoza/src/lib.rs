extern crate spinoza as spinoza_rs;

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use spinoza_rs::{
    circuit::{
        QuantumCircuit as QuantumCircuitRS, QuantumRegister as QuantumRegisterRS,
        QuantumTransformation as QuantumTransformationRS,
    },
    core::{reservoir_sampling, State},
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
    fn len(&self) -> usize {
        self.qr.len()
    }

    #[inline]
    fn update_shift(&mut self, shift: usize) {
        self.qr.update_shift(shift)
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

    pub fn get_statevector(&self) -> PyResult<PyState> {
        let temp = PyState {
            data: self.qc.state.clone(),
        };
        Ok(temp)
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
    pub fn execute(&mut self) {
        self.qc.execute();
    }
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
    qc.get_statevector()
}

#[pymodule]
fn spynoza(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(get_samples))?;
    m.add_wrapped(wrap_pyfunction!(show_table))?;
    m.add_wrapped(wrap_pyfunction!(run))?;
    m.add_class::<PyState>()?;
    m.add_class::<QuantumRegister>()?;
    m.add_class::<QuantumCircuit>()?;
    Ok(())
}
