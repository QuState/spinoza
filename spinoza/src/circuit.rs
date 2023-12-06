//! Abstractions for a quantum circuit
use std::{collections::HashSet, ops::Index};

use crate::{
    core::State,
    gates::{apply, c_apply, cc_apply, Gate},
    math::{pow2f, Float, PI},
    measurement::measure_qubit,
};

/// See <https://en.wikipedia.org/wiki/Quantum_register>
#[derive(Clone)]
pub struct QuantumRegister(pub Vec<usize>);

impl Index<usize> for QuantumRegister {
    type Output = usize;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl QuantumRegister {
    /// Create a new QuantumRegister
    pub fn new(size: usize) -> Self {
        assert!(size > 0);
        QuantumRegister((0..size).collect())
    }

    /// The length of the quantum register
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Update quantum register by shift
    #[inline]
    pub fn update_shift(&mut self, shift: usize) {
        self.0.iter_mut().for_each(|x| *x += shift);
    }

    /// Get the shift size for this register
    #[inline]
    pub fn get_shift(&self) -> usize {
        self[0]
    }
}

/// Control qubits
#[derive(Clone)]
pub enum Controls {
    /// No controls
    None,
    /// Single Control
    Single(usize),
    /// Multiple Controls
    Ones(Vec<usize>),

    /// Mixed Controls
    Mixed {
        /// Control qubits
        controls: Vec<usize>,
        /// Zeroes
        zeros: HashSet<usize>,
    },
}

impl Controls {
    fn from(&self, controls: &[usize], zeros: Option<HashSet<usize>>) -> Self {
        if let Some(zs) = zeros {
            Self::Mixed {
                controls: controls.to_vec(),
                zeros: zs,
            }
        } else if controls.is_empty() {
            Self::None
        } else if controls.len() == 1 {
            Self::Single(controls[0])
        } else {
            Self::Ones(controls.to_vec())
        }
    }

    fn unpack(&self) -> (Vec<usize>, HashSet<usize>) {
        match self {
            Self::None => (vec![], HashSet::new()),
            Self::Single(c0) => (vec![*c0], HashSet::new()),
            Self::Ones(controls) => (controls.clone(), HashSet::new()),
            Self::Mixed { controls, zeros } => (controls.clone(), zeros.clone()),
        }
    }

    fn new_with_control(&self, control: usize, shift: usize) -> Self {
        let (mut controls, mut zeros) = self.unpack();
        controls.iter_mut().for_each(|c| *c += shift);
        controls.push(control);

        if zeros.is_empty() {
            self.from(&controls, None)
        } else {
            zeros = zeros.iter().map(|z| z + shift).collect();
            self.from(&controls, Some(zeros))
        }
    }
}

/// QuantumTransformation to be applied to the State
#[derive(Clone)]
pub struct QuantumTransformation {
    /// The quantum logic gate
    pub gate: Gate,
    /// The target qubits
    pub target: usize,
    /// The control qubits
    pub controls: Controls,
}

struct QubitTracker {
    /// Used to keep track of the qubits that were already measured We can use a u64, since the
    /// number of qubits that one can simulate is very small. Hence, 64 bits suffices to keep
    /// track of all qubits.
    measured_qubits: u64,
    /// Used to keep track of the *values* of qubits that were already measured We can use a u64,
    /// since the number of qubits that one can simulate is very small, and the measured values are
    /// either 0 or 1. Hence, 64 bits suffices to keep track of all measurement values.
    measured_qubits_vals: u64,
}

impl QubitTracker {
    pub fn new() -> Self {
        Self {
            measured_qubits: 0,
            measured_qubits_vals: 0,
        }
    }

    fn is_qubit_measured(&self, target_qubit: usize) -> bool {
        ((self.measured_qubits >> target_qubit) & 1) == 1
    }

    fn get_qubit_measured_val(&mut self, target_qubit: usize) -> Option<u8> {
        if self.is_qubit_measured(target_qubit) {
            ((self.measured_qubits_vals & (1 << target_qubit)) >> target_qubit)
                .try_into()
                .ok()
        } else {
            None
        }
    }

    /// Set the given target qubit as measured
    fn set_measured_qubit(&mut self, target_qubit: usize) {
        self.measured_qubits |= 1 << target_qubit;
    }

    fn set_val_for_measured_qubit(&mut self, target_qubit: usize, value: u8) {
        self.measured_qubits_vals &= !(1 << target_qubit);
        self.measured_qubits_vals |= (value as u64) << target_qubit;
    }
}

/// A model of a Quantum circuit
/// See <https://en.wikipedia.org/wiki/Quantum_circuit>
pub struct QuantumCircuit {
    /// The list of operations to be applied to the State
    pub transformations: Vec<QuantumTransformation>,
    /// The Quantum State to which transformations are applied
    pub state: State,
    /// Tracks measured qubits for dynamic circuits
    qubit_tracker: QubitTracker,
    /// The sizes of the provided quantum registers
    pub quantum_registers_info: Vec<usize>,
}

impl QuantumCircuit {
    /// Create a new QuantumCircuit from multiple QuantumRegisters
    pub fn new(registers: &mut [&mut QuantumRegister]) -> Self {
        let mut bits = 0;

        let mut qr_sizes = Vec::with_capacity(registers.len());

        for r in registers.iter_mut() {
            r.update_shift(bits);
            qr_sizes.push(r.len());
            bits += r.len();
        }
        Self {
            transformations: Vec::new(),
            state: State::new(bits),
            qubit_tracker: QubitTracker::new(),
            quantum_registers_info: qr_sizes,
        }
    }

    /// Get a reference to the Quantum State
    pub fn get_statevector(&self) -> &State {
        &self.state
    }

    /// Invert this circuit
    pub fn inverse(&mut self) {
        self.transformations.reverse();
        self.transformations.iter_mut().for_each(|qt| {
            qt.gate = qt.gate.inverse();
        });
    }

    /// Measure a single qubit
    #[inline]
    pub fn measure(&mut self, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::M,
            target,
            controls: Controls::None,
        });
    }

    /// Add a SWAP gate that swaps two qubits
    #[inline]
    pub fn swap(&mut self, t0: usize, t1: usize) {
        self.add(QuantumTransformation {
            gate: Gate::SWAP((t0, t1)),
            target: 0,
            controls: Controls::None,
        });
    }

    /// Add the X gate for a given target to the list of QuantumTransformations
    #[inline]
    pub fn x(&mut self, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::X,
            target,
            controls: Controls::None,
        });
    }

    /// Add the Y gate for a given target to the list of QuantumTransformations
    #[inline]
    pub fn y(&mut self, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::Y,
            target,
            controls: Controls::None,
        });
    }

    /// Add the Rx gate for a given target to the list of QuantumTransformations
    #[inline]
    pub fn rx(&mut self, angle: Float, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::RX(angle),
            target,
            controls: Controls::None,
        });
    }

    /// Add the CX gate for a given target qubit and control qubit to the list of
    /// QuantumTransformations
    #[inline]
    pub fn cx(&mut self, control: usize, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::X,
            target,
            controls: Controls::Single(control),
        });
    }

    /// Add the CCX gate for a given target qubit and two control qubits to the list of
    /// QuantumTransformations
    #[inline]
    pub fn ccx(&mut self, control1: usize, control2: usize, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::X,
            target,
            controls: Controls::Ones(vec![control1, control2]),
        });
    }

    /// Add the Hadamard (H) gate for a given target qubit to the list of QuantumTransformations
    #[inline]
    pub fn h(&mut self, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::H,
            target,
            controls: Controls::None,
        });
    }

    /// Add Ry gate for a given target qubit to the list of QuantumTransformations
    #[inline]
    pub fn ry(&mut self, angle: Float, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::RY(angle),
            target,
            controls: Controls::None,
        });
    }

    /// Add CRy gate for a given target qubit and a given control qubit to the list of
    /// QuantumTransformations
    #[inline]
    pub fn cry(&mut self, angle: Float, control: usize, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::RY(angle),
            target,
            controls: Controls::Single(control),
        });
    }

    /// Add Phase (P) gate for a given target qubit to the list of QuantumTransformations
    #[inline]
    pub fn p(&mut self, angle: Float, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::P(angle),
            target,
            controls: Controls::None,
        });
    }

    /// Add the Controlled Phase (CP) gate for a given target qubit and a given control qubit to
    /// the list of QuantumTransformations
    #[inline]
    pub fn cp(&mut self, angle: Float, control: usize, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::P(angle),
            target,
            controls: Controls::Single(control),
        });
    }

    /// Add the Controlled Y gate for a given target qubit and a given control qubit to the list of
    /// QuantumTransformations
    #[inline]
    pub fn cy(&mut self, control: usize, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::Y,
            target,
            controls: Controls::Single(control),
        });
    }

    /// Add the Controlled Rx gate for a given target qubit and a given control qubit to the list of
    /// QuantumTransformations
    #[inline]
    pub fn crx(&mut self, angle: Float, control: usize, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::RX(angle),
            target,
            controls: Controls::Single(control),
        });
    }

    /// Add Z gate for a given target qubit to the list of QuantumTransformations
    #[inline]
    pub fn z(&mut self, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::Z,
            target,
            controls: Controls::None,
        });
    }

    /// Add Rz gate for a given target qubit to the list of QuantumTransformations
    #[inline]
    pub fn rz(&mut self, angle: Float, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::RZ(angle),
            target,
            controls: Controls::None,
        });
    }

    /// Add the U gate for a given target to the list of QuantumTransformations
    #[inline]
    pub fn u(&mut self, theta: Float, phi: Float, lambda: Float, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::U((theta, phi, lambda)),
            target,
            controls: Controls::None,
        });
    }

    /// Add all transformations for an inverse Quantum Fourier Transform to the list of QuantumTransformations
    #[inline]
    pub fn iqft(&mut self, targets: &[usize]) {
        for j in (0..targets.len()).rev() {
            self.h(targets[j]);
            for k in (0..j).rev() {
                self.cp(-PI / pow2f(j - k), targets[j], targets[k]);
            }
        }
    }

    /// Append a QuantumCircuit to the given register of *this* QuantumCircuit
    pub fn append(&mut self, circuit: &QuantumCircuit, reg: &QuantumRegister) {
        assert!(reg.len() == circuit.quantum_registers_info.iter().sum());
        for tr in circuit.transformations.iter() {
            self.add(QuantumTransformation {
                gate: tr.gate,
                target: reg.get_shift() + tr.target,
                controls: tr.controls.clone(),
            });
        }
    }

    /// Append a QuantumCircuit to the given register of *this* QuantumCircuit
    pub fn c_append(&mut self, circuit: &QuantumCircuit, c: usize, reg: &QuantumRegister) {
        assert!(!std::ops::Range {
            start: reg.get_shift(),
            end: reg.get_shift() + reg.len()
        }
        .contains(&c));
        for tr in circuit.transformations.iter() {
            self.add(QuantumTransformation {
                gate: tr.gate,
                target: reg.get_shift() + tr.target,
                controls: tr.controls.new_with_control(c, reg.get_shift()),
            });
        }
    }

    /// Append a QuantumCircuit to the given register of *this* QuantumCircuit
    pub fn mc_append(
        &mut self,
        circuit: &QuantumCircuit,
        controls: &[usize],
        reg: &QuantumRegister,
    ) {
        let hashed_controls: HashSet<_> = controls.iter().copied().collect();

        assert_eq!(controls.len(), hashed_controls.len());
        let range = std::ops::Range {
            start: reg.get_shift(),
            end: reg.get_shift() + reg.len(),
        };

        controls.iter().for_each(|c| {
            if range.contains(c) {
                panic!(
                    "control {} should not be in: Range(start: {} end: {})",
                    c, range.start, range.end
                );
            }
        });

        for control in controls.iter() {
            for tr in circuit.transformations.iter() {
                self.add(QuantumTransformation {
                    gate: tr.gate,
                    target: reg.get_shift() + tr.target,
                    controls: tr.controls.new_with_control(*control, reg.get_shift()),
                });
            }
        }
    }

    /// Add a given `QuantumTransformation` to the list of transformations
    #[inline]
    pub fn add(&mut self, transformation: QuantumTransformation) {
        self.transformations.push(transformation)
    }

    /// Run the list of transformations against the State
    pub fn execute(&mut self) {
        for tr in self.transformations.drain(..) {
            match (&tr.controls, tr.gate) {
                (Controls::None, Gate::M) => {
                    if !self.qubit_tracker.is_qubit_measured(tr.target) {
                        let value = measure_qubit(&mut self.state, tr.target, true, None);
                        self.qubit_tracker.set_measured_qubit(tr.target);
                        self.qubit_tracker
                            .set_val_for_measured_qubit(tr.target, value);
                    }
                }
                (Controls::None, gate) => {
                    apply(gate, &mut self.state, tr.target);
                }
                (Controls::Single(control), gate) => {
                    if let Some(c_bit) = self.qubit_tracker.get_qubit_measured_val(*control) {
                        if c_bit == 1 {
                            apply(gate, &mut self.state, tr.target);
                        }
                    } else {
                        c_apply(gate, &mut self.state, *control, tr.target);
                    }
                }
                (Controls::Ones(controls), Gate::X) => {
                    cc_apply(
                        Gate::X,
                        &mut self.state,
                        controls[0],
                        controls[1],
                        tr.target,
                    );
                }
                _ => {
                    todo!();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::to_table;
    use crate::{
        math::modulus,
        utils::{assert_float_closeness, gen_random_state, swap},
    };

    #[test]
    fn register_shift() {
        const N: usize = 4;
        let mut qr = QuantumRegister::new(N);
        assert_eq!(qr.get_shift(), 0);

        qr.update_shift(4);
        assert_eq!(qr.get_shift(), 4);
    }

    #[test]
    fn value_encoding() {
        let v = 2.4;
        let n = 3;

        let mut q = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut [&mut q]);

        for i in 0..n {
            qc.h(i)
        }

        for i in 0..n {
            qc.p((2 as Float) * PI / pow2f(i + 1) * v, i)
        }

        let targets: Vec<usize> = (0..n).rev().collect();
        qc.iqft(&targets);
        qc.execute();
    }

    #[test]
    fn z_gate() {
        let n = 2;
        let mut q = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut [&mut q]);

        for i in 0..n {
            qc.h(i)
        }

        qc.z(0);

        qc.execute();
    }

    #[test]
    fn crx() {
        let n = 3;
        let mut q = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut [&mut q]);

        for i in 0..n {
            qc.h(i)
        }

        let mut t = 0;
        while t < n - 1 {
            qc.crx(3.043, t, t + 1);
            t += 2;
        }

        qc.execute();
    }

    #[test]
    fn cy() {
        let n = 3;
        let mut q = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut [&mut q]);

        for i in 0..n {
            qc.h(i)
        }

        let mut t = 0;
        while t < n - 1 {
            qc.cy(t, t + 1);
            t += 2;
        }

        qc.execute();
    }

    #[test]
    fn ccx() {
        let n = 3;
        let mut q = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut [&mut q]);

        for i in 0..n - 1 {
            qc.h(i)
        }

        qc.ccx(0, 1, 2);

        qc.execute();
        let _state = qc.get_statevector();
    }

    #[test]
    fn x_gate() {
        let n = 2;
        let mut q = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut [&mut q]);

        for i in 0..n {
            qc.h(i)
        }

        qc.rx(PI / 2.0, 0);
        qc.execute();
    }

    #[test]
    fn all_gates_as_transformations() {
        const N: usize = 17;
        let mut q = QuantumRegister::new(N);
        let mut qc = QuantumCircuit::new(&mut [&mut q]);

        for t in 0..N {
            qc.h(t)
        }

        qc.x(0);
        qc.y(1);
        qc.z(2);
        qc.p(PI, 3);
        qc.cp(PI, 3, 4);
        qc.rx(PI, 5);
        qc.ry(PI, 6);
        qc.rz(PI, 7);
        qc.u(PI, PI, PI, 8);
        qc.cy(9, 10);
        qc.crx(PI, 11, 12);
        qc.cry(PI, 13, 14);
        qc.execute();

        let mut state = State::new(N);
        for t in 0..N {
            apply(Gate::H, &mut state, t);
        }

        apply(Gate::X, &mut state, 0);
        apply(Gate::Y, &mut state, 1);
        apply(Gate::Z, &mut state, 2);
        apply(Gate::P(PI), &mut state, 3);
        c_apply(Gate::P(PI), &mut state, 3, 4);
        apply(Gate::RX(PI), &mut state, 5);
        apply(Gate::RY(PI), &mut state, 6);
        apply(Gate::RZ(PI), &mut state, 7);
        apply(Gate::U((PI, PI, PI)), &mut state, 8);
        c_apply(Gate::Y, &mut state, 9, 10);
        c_apply(Gate::RX(PI), &mut state, 11, 12);
        c_apply(Gate::RY(PI), &mut state, 13, 14);

        state
            .reals
            .iter()
            .zip(state.imags.iter())
            .zip(qc.state.reals.iter())
            .zip(qc.state.imags.iter())
            .for_each(|(((s_re, s_im), qc_re), qc_im)| {
                assert_float_closeness(*qc_re, *s_re, 0.001);
                assert_float_closeness(*qc_im, *s_im, 0.001);
            });
    }

    #[test]
    fn measure() {
        const N: usize = 21;
        let state = gen_random_state(N);

        let sum = state
            .reals
            .iter()
            .zip(state.imags.iter())
            .map(|(re, im)| modulus(*re, *im).powi(2))
            .sum();

        // Make sure the generated random state is sound
        assert_float_closeness(sum, 1.0, 0.001);

        let mut qc = QuantumCircuit {
            state,
            transformations: Vec::new(),
            qubit_tracker: QubitTracker::new(),
            quantum_registers_info: Vec::new(),
        };

        // Add Measure gates for all the qubits
        for target in 0..N {
            qc.measure(target);
        }

        // Execute the circuit
        qc.execute();

        // Allocate stack storage of measured qubit values
        let mut measured_vals = [0; N];

        // Now collect the measured values
        for target in 0..N {
            let val = qc
                .qubit_tracker
                .get_qubit_measured_val(target)
                .expect("qubit: {target} should be measured");
            measured_vals[target] = val;
        }

        // Now we need to measure again...

        // Add Measure gates for all the qubits
        for target in 0..N {
            qc.measure(target);
        }

        // Execute the circuit
        qc.execute();

        // Now check that the new measured values are the same as what we got before
        for target in 0..N {
            assert!(
                qc.qubit_tracker.is_qubit_measured(target),
                "qubit {target} was already measured, but it wasn't marked as measured"
            );
            assert_eq!(
                measured_vals[target],
                qc.qubit_tracker
                    .get_qubit_measured_val(target)
                    .expect("qubit: {target} should be measured")
            );
        }
    }

    #[test]
    fn swap_all_qubits() {
        const N: usize = 9;
        let mut qc = QuantumCircuit {
            state: gen_random_state(N),
            transformations: Vec::new(),
            qubit_tracker: QubitTracker::new(),
            quantum_registers_info: Vec::new(),
        };

        let sum = qc
            .state
            .reals
            .iter()
            .zip(qc.state.imags.iter())
            .map(|(re, im)| modulus(*re, *im).powi(2))
            .sum();

        // Make sure the generated random state is sound
        assert_float_closeness(sum, 1.0, 0.001);

        let mut state = qc.state.clone();

        // Add swap gates, for pairs of qubits, to the circuit. Simultaneously, directly apply swap
        // to the lone state (using the 3 CX gate implementation)
        for i in 0..(N >> 1) {
            qc.swap(i, N - 1 - i);
            swap(&mut state, i, N - 1 - i);
        }

        qc.execute();

        assert_eq!(qc.state.n, state.n);
        assert_eq!(qc.state.reals, state.reals);
        assert_eq!(qc.state.imags, state.imags);
    }

    #[test]
    fn inverse_iqft() {
        const N: usize = 2;
        let original_state = gen_random_state(N);
        let mut qc1 = QuantumCircuit {
            state: original_state.clone(),
            transformations: Vec::new(),
            qubit_tracker: QubitTracker::new(),
            quantum_registers_info: Vec::new(),
        };

        // First we apply iqft to all qubits
        let targets: Vec<_> = (0..N).rev().collect();
        qc1.iqft(&targets);
        qc1.execute();

        // Next, we apply inverse of iqft to all qubits
        qc1.iqft(&targets);
        qc1.inverse();
        qc1.execute();

        // Check that after applying iqft and iqft inverse, we get the original state back
        original_state
            .reals
            .iter()
            .zip(original_state.imags.iter())
            .zip(qc1.state.reals.iter())
            .zip(qc1.state.imags.iter())
            .for_each(|(((os_re, os_im), qc1_re), qc1_im)| {
                assert_float_closeness(*qc1_re, *os_re, 0.001);
                assert_float_closeness(*qc1_im, *os_im, 0.001);
            });
    }

    #[test]
    fn inverse() {
        const N: usize = 2;
        let mut qc1 = QuantumCircuit {
            state: State::new(N),
            transformations: Vec::new(),
            qubit_tracker: QubitTracker::new(),
            quantum_registers_info: Vec::new(),
        };

        qc1.h(0);
        qc1.p(PI / 4.0, 1);
        qc1.inverse();
        qc1.execute();

        let mut qc2 = QuantumCircuit {
            state: State::new(N),
            transformations: Vec::new(),
            qubit_tracker: QubitTracker::new(),
            quantum_registers_info: Vec::new(),
        };
        qc2.p(-(PI / 4.0), 1);
        qc2.h(0);
        qc2.execute();

        assert_eq!(qc1.state.reals, qc2.state.reals);
        assert_eq!(qc1.state.imags, qc2.state.imags);
    }

    // Helper function for testing QuantumCircuit *append methods
    fn gate_to_single_qubit_circuit(gate: Gate) -> QuantumCircuit {
        let mut qr = QuantumRegister::new(1);
        let mut qc = QuantumCircuit::new(&mut [&mut qr]);
        qc.add(QuantumTransformation {
            gate,
            target: 0,
            controls: Controls::None,
        });
        qc
    }

    // Helper function for testing `QuantumCircuit`'s `append` method
    fn gate_to_circuit(gate: Gate, n: usize, target: usize) -> QuantumCircuit {
        let mut qr = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut [&mut qr]);
        qc.add(QuantumTransformation {
            gate,
            target,
            controls: Controls::None,
        });
        qc
    }

    // Helper function for creating an IQFT circuit for testing `QuantumCicuit`'s
    // `append` method
    fn iqft_circuit(n: usize) -> QuantumCircuit {
        let mut iqft_qr = QuantumRegister::new(n);
        let mut iqft_qc = QuantumCircuit::new(&mut [&mut iqft_qr]);
        let targets: Vec<usize> = (0..n).rev().collect();
        iqft_qc.iqft(&targets);
        iqft_qc
    }

    // Helper function for creating an IQFT circuit for testing `QuantumCircuit`'s
    // `c_append` or `mc_append` method
    fn iqft_circuit_from_controlled_append(n: usize, multi_control: bool) -> QuantumCircuit {
        let mut iqft_qr = QuantumRegister::new(n);
        let mut iqft_qc = QuantumCircuit::new(&mut [&mut iqft_qr]);
        let targets: Vec<usize> = (0..n).rev().collect();

        for j in (0..targets.len()).rev() {
            iqft_qc.append(&gate_to_circuit(Gate::H, n, targets[j]), &iqft_qr);
            for k in (0..j).rev() {
                let mut qr = QuantumRegister::new(1);
                qr.update_shift(targets[k]);
                if multi_control {
                    iqft_qc.mc_append(
                        &gate_to_single_qubit_circuit(Gate::P(-PI / pow2f(j - k))),
                        &[targets[j]],
                        &qr,
                    );
                } else {
                    iqft_qc.c_append(
                        &gate_to_single_qubit_circuit(Gate::P(-PI / pow2f(j - k))),
                        targets[j],
                        &qr,
                    );
                }
            }
        }
        iqft_qc
    }

    #[test]
    fn append() {
        let mut qr0 = QuantumRegister::new(1);
        let mut qr1 = QuantumRegister::new(1);
        let mut qc = QuantumCircuit::new(&mut [&mut qr0, &mut qr1]);
        qc.append(&gate_to_single_qubit_circuit(Gate::H), &qr0);
        qc.append(&gate_to_single_qubit_circuit(Gate::H), &qr1);
        qc.execute();

        qc.state
            .reals
            .iter()
            .zip(qc.state.imags.iter())
            .for_each(|(z_re, z_im)| {
                assert_float_closeness(*z_re, 0.5, 0.0001);
                assert_float_closeness(*z_im, 0.0, 0.0001);
            });
        println!("{}", to_table(&qc.state));
    }

    #[test]
    fn append_value_encoding() {
        let n = 3;
        let v = 4.0;
        let mut qr = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut [&mut qr]);

        for t in 0..n {
            qc.append(&gate_to_circuit(Gate::H, n, t), &qr);
        }
        for t in 0..n {
            qc.append(
                &gate_to_circuit(Gate::P(2.0 * PI / (pow2f(t + 1)) * v), n, t),
                &qr,
            );
        }

        let iqft_qc = iqft_circuit(n);
        qc.append(&iqft_qc, &qr);
        qc.execute();

        let encoded_integer = v as usize;

        qc.state
            .reals
            .iter()
            .zip(qc.state.imags.iter())
            .enumerate()
            .for_each(|(i, (z_re, z_im))| {
                if i == encoded_integer {
                    assert_float_closeness(*z_re, 1.0, 0.0001);
                    assert_float_closeness(*z_im, 0.0, 0.0001);
                } else {
                    assert_float_closeness(*z_re, 0.0, 0.0001);
                    assert_float_closeness(*z_im, 0.0, 0.0001);
                }
            });
        println!("{}", to_table(&qc.state));
    }

    #[test]
    fn c_append_value_encoding() {
        let n = 3;
        let v = 4.0;
        let mut qr = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut [&mut qr]);

        for t in 0..n {
            qc.append(&gate_to_circuit(Gate::H, n, t), &qr);
        }
        for t in 0..n {
            qc.append(
                &gate_to_circuit(Gate::P(2.0 * PI / (pow2f(t + 1)) * v), n, t),
                &qr,
            );
        }

        let iqft_qc = iqft_circuit_from_controlled_append(n, false);
        qc.append(&iqft_qc, &qr);
        qc.execute();
        let encoded_integer = v as usize;

        qc.state
            .reals
            .iter()
            .zip(qc.state.imags.iter())
            .enumerate()
            .for_each(|(i, (z_re, z_im))| {
                if i == encoded_integer {
                    assert_float_closeness(*z_re, 1.0, 0.0001);
                    assert_float_closeness(*z_im, 0.0, 0.0001);
                } else {
                    assert_float_closeness(*z_re, 0.0, 0.0001);
                    assert_float_closeness(*z_im, 0.0, 0.0001);
                }
            });
        println!("{}", to_table(&qc.state));
    }

    #[test]
    fn mc_append_value_encoding() {
        let n = 3;
        let v = 4.0;
        let mut qr = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut [&mut qr]);

        for t in 0..n {
            qc.append(&gate_to_circuit(Gate::H, n, t), &qr);
        }
        for t in 0..n {
            qc.append(
                &gate_to_circuit(Gate::P(2.0 * PI / (pow2f(t + 1)) * v), n, t),
                &qr,
            );
        }

        let iqft_qc = iqft_circuit_from_controlled_append(n, true);
        qc.append(&iqft_qc, &qr);
        qc.execute();

        let encoded_integer = v as usize;

        qc.state
            .reals
            .iter()
            .zip(qc.state.imags.iter())
            .enumerate()
            .for_each(|(i, (z_re, z_im))| {
                if i == encoded_integer {
                    assert_float_closeness(*z_re, 1.0, 0.0001);
                    assert_float_closeness(*z_im, 0.0, 0.0001);
                } else {
                    assert_float_closeness(*z_re, 0.0, 0.0001);
                    assert_float_closeness(*z_im, 0.0, 0.0001);
                }
            });
        println!("{}", to_table(&qc.state));
    }
}
