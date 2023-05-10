use crate::{
    core::State,
    gates::{apply, c_apply, cc_apply, Gate},
    math::{pow2f, Float, PI},
};
use std::{collections::HashSet, ops::Index};

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
    pub fn new(size: usize) -> Self {
        QuantumRegister((0..size).collect())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn update_shift(&mut self, shift: usize) {
        self.0 = (shift..shift + self.len()).collect();
    }
}

#[derive(Clone)]
pub enum Controls {
    None,
    Single(usize),
    Ones(Vec<usize>),

    #[allow(dead_code)]
    Mixed {
        controls: Vec<usize>,
        zeros: HashSet<usize>,
    },
}

#[derive(Clone)]
pub struct QuantumTransformation {
    pub gate: Gate,
    pub target: usize,
    pub controls: Controls,
}

pub struct QuantumCircuit {
    transformations: Vec<QuantumTransformation>,
    pub state: State,
}

impl QuantumCircuit {
    pub fn new_multi(registers: &mut [&mut QuantumRegister]) -> Self {
        let mut bits = 0;

        for r in registers {
            r.update_shift(bits);
            bits += r.len();
        }
        QuantumCircuit {
            transformations: Vec::new(),
            state: State::new(bits),
        }
    }

    pub fn new(r: &mut QuantumRegister) -> Self {
        QuantumCircuit::new_multi(&mut [r])
    }

    pub fn new2(r0: &mut QuantumRegister, r1: &mut QuantumRegister) -> Self {
        QuantumCircuit::new_multi(&mut [r0, r1])
    }

    pub fn get_statevector(&self) -> &State {
        &self.state
    }

    #[inline]
    pub fn x(&mut self, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::X,
            target,
            controls: Controls::None,
        });
    }

    #[inline]
    pub fn rx(&mut self, angle: Float, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::RX(angle),
            target,
            controls: Controls::None,
        });
    }

    #[inline]
    pub fn cx(&mut self, control: usize, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::X,
            target,
            controls: Controls::Single(control),
        });
    }

    #[inline]
    pub fn ccx(&mut self, control1: usize, control2: usize, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::X,
            target,
            controls: Controls::Ones(vec![control1, control2]),
        });
    }

    #[inline]
    pub fn h(&mut self, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::H,
            target,
            controls: Controls::None,
        });
    }

    #[inline]
    pub fn ry(&mut self, angle: Float, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::RY(angle),
            target,
            controls: Controls::None,
        });
    }

    #[inline]
    pub fn cry(&mut self, angle: Float, control: usize, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::RY(angle),
            target,
            controls: Controls::Single(control),
        });
    }

    #[inline]
    pub fn p(&mut self, angle: Float, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::P(angle),
            target,
            controls: Controls::None,
        });
    }

    #[inline]
    pub fn cp(&mut self, angle: Float, control: usize, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::P(angle),
            target,
            controls: Controls::Single(control),
        });
    }

    #[inline]
    pub fn z(&mut self, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::Z,
            target,
            controls: Controls::None,
        });
    }

    #[inline]
    pub fn rz(&mut self, angle: Float, target: usize) {
        self.add(QuantumTransformation {
            gate: Gate::RZ(angle),
            target,
            controls: Controls::None,
        });
    }

    #[inline]
    pub fn iqft(&mut self, targets: &[usize]) {
        for j in (0..targets.len()).rev() {
            self.h(targets[j]);
            for k in (0..j).rev() {
                self.cp(-PI / pow2f(j - k), targets[j], targets[k]);
            }
        }
    }

    #[inline]
    pub fn add(&mut self, transformation: QuantumTransformation) {
        self.transformations.push(transformation)
    }

    pub fn execute(&mut self) {
        for tr in self.transformations.drain(..) {
            match (&tr.controls, tr.gate) {
                (Controls::None, Gate::RZ(theta)) => {
                    apply(Gate::RZ(theta), &mut self.state, tr.target);
                }
                (Controls::None, Gate::RX(theta)) => {
                    apply(Gate::RX(theta), &mut self.state, tr.target);
                }
                (Controls::None, Gate::X) => {
                    apply(Gate::X, &mut self.state, tr.target);
                }
                (Controls::None, Gate::RY(theta)) => {
                    apply(Gate::RY(theta), &mut self.state, tr.target);
                }
                (Controls::None, Gate::Z) => {
                    apply(Gate::Z, &mut self.state, tr.target);
                }
                (Controls::None, Gate::H) => {
                    apply(Gate::H, &mut self.state, tr.target);
                }
                (Controls::None, Gate::P(theta)) => {
                    apply(Gate::P(theta), &mut self.state, tr.target);
                }
                (Controls::Single(control), Gate::X) => {
                    c_apply(Gate::X, &mut self.state, *control, tr.target);
                }
                (Controls::Single(control), Gate::P(theta)) => {
                    c_apply(Gate::P(theta), &mut self.state, *control, tr.target);
                }
                (Controls::Single(control), Gate::RY(theta)) => {
                    c_apply(Gate::RY(theta), &mut self.state, *control, tr.target);
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

    #[test]
    fn circuit() {
        let n = 2;
        let mut q = QuantumRegister::new(n);
        let mut a = QuantumRegister::new(1);
        let mut qc = QuantumCircuit::new2(&mut q, &mut a);

        for i in 0..n {
            qc.ry(PI / 2.0, q[i]);
        }
        qc.cry(PI / 2.0, q[0], q[1]);
        qc.p(PI / 2.0, q[0]);
        qc.cp(PI / 2.0, q[0], q[1]);
        qc.h(a[0]);

        qc.execute();
    }

    #[test]
    fn value_encoding() {
        let v = 2.4;
        let n = 3;

        let mut q = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut q);

        for i in 0..n {
            qc.h(q[i])
        }

        for i in 0..n {
            qc.p((2 as Float) * PI / pow2f(i + 1) * v, q[i])
        }

        let targets: Vec<usize> = (0..n).rev().collect();
        qc.iqft(&targets);
        qc.execute();
    }

    #[test]
    fn z_gate() {
        let n = 2;
        let mut q = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut q);

        for i in 0..n {
            qc.h(q[i])
        }

        qc.z(q[0]);

        qc.execute();
    }

    #[test]
    fn ccx() {
        let n = 3;
        let mut q = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut q);

        for i in 0..n - 1 {
            qc.h(q[i])
        }

        qc.ccx(q[0], q[1], q[2]);

        qc.execute();
        let _state = qc.get_statevector();
    }

    #[test]
    fn x_gate() {
        let n = 2;
        let mut q = QuantumRegister::new(n);
        let mut qc = QuantumCircuit::new(&mut q);

        for i in 0..n {
            qc.h(q[i])
        }

        qc.rx(PI / 2.0, q[0]);
        qc.execute();
    }
}
