//! Functionality for applying large 2^n * 2^n matrices to the state
//! Ideally, this should be a last resort
use crate::{core::State, gates::Gate, math::Float};
use rayon::prelude::*;
use std::fmt;
use std::fmt::Formatter;

/// A representation of a Unitary Matrix
#[derive(Clone)]
pub struct Unitary {
    pub(crate) reals: Vec<Float>,
    pub(crate) imags: Vec<Float>,
    /// The number of rows in the matrix
    pub height: usize,
    /// The number of columns in the matrix
    pub width: usize,
}

impl fmt::Display for Unitary {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.reals
            .chunks_exact(self.width)
            .zip(self.imags.chunks_exact(self.width))
            .for_each(|(re, im)| {
                re.iter().zip(im.iter()).for_each(|(z_re, z_im)| {
                    write!(f, "{z_re}+i{z_im} ").unwrap();
                });
                writeln!(f).unwrap();
            });
        Ok(())
    }
}

impl Unitary {
    // TODO(saveliy): look into using strategy 2 here
    /// Construct a Unitary from a single qubit gate
    pub fn from_single_qubit_gate(state: &State, gate: Gate, target: usize) -> Self {
        let g = gate.to_matrix();
        let num_pairs = state.len() >> 1;
        let distance = 1 << target;

        let width = state.len();
        let height = state.len();
        let mut reals = vec![0.0; height * width];
        let mut imags = vec![0.0; height * width];

        for i in 0..num_pairs {
            let s0 = i + ((i >> target) << target);
            let s1 = s0 + distance;

            reals[width * s0 + s0] = g[0].re;
            reals[width * s0 + s1] = g[1].re;
            reals[width * s1 + s0] = g[2].re;
            reals[width * s1 + s1] = g[3].re;

            imags[width * s0 + s0] = g[0].im;
            imags[width * s0 + s1] = g[1].im;
            imags[width * s1 + s0] = g[2].im;
            imags[width * s1 + s1] = g[3].im;
        }

        Self {
            reals,
            imags,
            height,
            width,
        }
    }

    /// Return the conjugate transpose of this `Unitary`, in-place
    pub fn conj_t(&mut self) {
        self.imags.par_iter_mut().for_each(|z_im| {
            *z_im = -(*z_im);
        });

        for i in 0..self.height {
            for j in i + 1..self.width {
                self.reals.swap(i * self.width + j, j * self.width + i);
                self.imags.swap(i * self.width + j, j * self.width + i);
            }
        }
    }

    /// Multiply this unitary matrix by another unitary matrix
    pub fn multiply(&self, other: &Unitary) -> Self {
        assert_eq!(self.width, other.height);

        let mut result_reals = vec![0.0; self.height * other.width];
        let mut result_imags = vec![0.0; self.height * other.width];

        // self.reals
        //     .chunks_exact(self.width)
        //     .zip(self.imags.chunks_exact(self.width))
        //     .enumerate()
        //     .map(|(i, (z_re, z_im))| {
        //         for k in 0..other.height {
        //
        //         }
        //     });

        for i in 0..self.height {
            for j in 0..other.width {
                let mut sum_real = 0.0;
                let mut sum_imag = 0.0;

                for k in 0..self.width {
                    let a_real = self.reals[i * self.width + k];
                    let a_imag = self.imags[i * self.width + k];
                    let b_real = other.reals[k * other.width + j];
                    let b_imag = other.imags[k * other.width + j];

                    sum_real += a_real * b_real - a_imag * b_imag;
                    sum_imag += a_real * b_imag + a_imag * b_real;
                }

                result_reals[i * other.width + j] = sum_real;
                result_imags[i * other.width + j] = sum_imag;
            }
        }

        Unitary {
            reals: result_reals,
            imags: result_imags,
            height: self.height,
            width: other.width,
        }
    }
}

/// Applies a unitary matrix to the Quantum State Vector
pub fn apply_unitary(state: &State, unitary: &Unitary) -> State {
    assert!(state.len() == unitary.width && state.len() == unitary.height);
    let chunk_size = unitary.width;

    let mut reals = Vec::with_capacity(state.len());
    let mut imags = Vec::with_capacity(state.len());

    unitary
        .reals
        .chunks_exact(chunk_size)
        .zip(unitary.imags.chunks_exact(chunk_size))
        .for_each(|(row_reals, row_imags)| {
            let mut dot_prod_re = 0.0;
            let mut dot_prod_im = 0.0;
            row_reals
                .iter()
                .zip(row_imags.iter())
                .zip(state.reals.iter())
                .zip(state.imags.iter())
                .for_each(|(((a, b), s_re), s_im)| {
                    dot_prod_re += *a * s_re - *b * s_im;
                    dot_prod_im += *a * s_im + *b * s_re;
                });
            reals.push(dot_prod_re);
            imags.push(dot_prod_im);
        });
    State {
        reals,
        imags,
        n: state.n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_float_closeness, gen_random_state};
    use crate::{
        gates::{apply, Gate},
        math::SQRT_ONE_HALF,
    };

    #[test]
    fn test_hxi_from_single_qubit_gate() {
        let mut reals = vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0,
        ];
        reals.iter_mut().for_each(|a| *a *= SQRT_ONE_HALF);

        let imags = vec![0.0; reals.len()];

        let u1 = Unitary {
            reals,
            imags,
            height: 4,
            width: 4,
        };

        let state = State::new(2);
        let u2 = Unitary::from_single_qubit_gate(&state, Gate::H, 1);
        assert_eq!(u1.height, u2.height);
        assert_eq!(u1.width, u2.width);
        assert_eq!(u1.reals, u2.reals);
        assert_eq!(u1.imags, u2.imags);
    }

    #[test]
    fn test_hxi() {
        let n = 2;
        let state = State::new(n);
        let u = Unitary::from_single_qubit_gate(&state, Gate::H, 0);

        let s = apply_unitary(&state, &u);

        let mut s1 = State::new(n);
        apply(Gate::H, &mut s1, 0);

        assert_eq!(s.n, s1.n);
        assert_eq!(s.reals, s1.reals);
        assert_eq!(s.imags, s1.imags);
    }

    #[test]
    fn test_x() {
        let n = 2;
        let state = State::new(n);
        let u = Unitary::from_single_qubit_gate(&state, Gate::X, 0);

        let s = apply_unitary(&state, &u);

        let mut s1 = State::new(n);
        apply(Gate::X, &mut s1, 0);

        assert_eq!(s.n, s1.n);
        assert_eq!(s.reals, s1.reals);
        assert_eq!(s.imags, s1.imags);
    }

    #[test]
    fn test_y() {
        let n = 2;
        let state = State::new(n);
        let u = Unitary::from_single_qubit_gate(&state, Gate::Y, 0);

        let s = apply_unitary(&state, &u);

        let mut s1 = State::new(n);
        apply(Gate::Y, &mut s1, 0);

        assert_eq!(s.n, s1.n);
        assert_eq!(s.reals, s1.reals);
        assert_eq!(s.imags, s1.imags);
    }

    #[test]
    fn test_z() {
        let n = 2;
        let state = State::new(n);
        let u = Unitary::from_single_qubit_gate(&state, Gate::Z, 0);

        let s = apply_unitary(&state, &u);

        let mut s1 = State::new(n);
        apply(Gate::Z, &mut s1, 0);

        assert_eq!(s.n, s1.n);
        assert_eq!(s.reals, s1.reals);
        assert_eq!(s.imags, s1.imags);
    }

    #[test]
    fn test_p() {
        let n = 2;
        let state = State::new(n);
        let u = Unitary::from_single_qubit_gate(&state, Gate::P(3.0), 0);

        let s = apply_unitary(&state, &u);

        let mut s1 = State::new(n);
        apply(Gate::P(3.0), &mut s1, 0);

        assert_eq!(s.n, s1.n);
        assert_eq!(s.reals, s1.reals);
        assert_eq!(s.imags, s1.imags);
    }

    #[test]
    fn test_rx() {
        let n = 2;
        let state = State::new(n);
        let u = Unitary::from_single_qubit_gate(&state, Gate::RX(3.0), 0);

        let s = apply_unitary(&state, &u);

        let mut s1 = State::new(n);
        apply(Gate::RX(3.0), &mut s1, 0);

        assert_eq!(s.n, s1.n);
        assert_eq!(s.reals, s1.reals);
        assert_eq!(s.imags, s1.imags);
    }

    #[test]
    fn test_ry() {
        let n = 2;
        let state = State::new(n);
        let u = Unitary::from_single_qubit_gate(&state, Gate::RY(3.0), 0);

        let s = apply_unitary(&state, &u);

        let mut s1 = State::new(n);
        apply(Gate::RY(3.0), &mut s1, 0);

        assert_eq!(s.n, s1.n);
        assert_eq!(s.reals, s1.reals);
        assert_eq!(s.imags, s1.imags);
    }

    #[test]
    fn test_rz() {
        let n = 2;
        let state = State::new(n);
        let u = Unitary::from_single_qubit_gate(&state, Gate::RZ(3.0), 0);

        let s = apply_unitary(&state, &u);

        let mut s1 = State::new(n);
        apply(Gate::RZ(3.0), &mut s1, 0);

        assert_eq!(s.n, s1.n);
        assert_eq!(s.reals, s1.reals);
        assert_eq!(s.imags, s1.imags);
    }

    #[test]
    fn test_u() {
        let n = 2;
        let state = State::new(n);
        let u = Unitary::from_single_qubit_gate(&state, Gate::U(1.0, 2.0, 3.0), 0);

        let s = apply_unitary(&state, &u);

        let mut s1 = State::new(n);
        apply(Gate::U(1.0, 2.0, 3.0), &mut s1, 0);

        assert_eq!(s.n, s1.n);
        assert_eq!(s.reals, s1.reals);
        assert_eq!(s.imags, s1.imags);
    }

    #[test]
    fn display() {
        const N: usize = 1;
        let state = gen_random_state(N);
        let u = Unitary::from_single_qubit_gate(&state, Gate::X, 0);

        let x_gate_as_str = "0+i0 1+i0 \n1+i0 0+i0 \n".to_string();
        let u_as_str = u.to_string();
        assert_eq!(u_as_str, x_gate_as_str);
    }

    #[test]
    fn conjugate_transpose() {
        const N: usize = 2;
        let state = State::new(N);
        let u = Unitary::from_single_qubit_gate(&state, Gate::H, 0);
        let mut u_ct = Unitary::from_single_qubit_gate(&state, Gate::H, 0);
        u_ct.conj_t();

        let identity = u_ct.multiply(&u);

        for i in 0..u.height {
            for j in 0..u.width {
                if i == j {
                    assert_float_closeness(identity.reals[i * identity.width + j], 1.0, 0.00001);
                    assert_float_closeness(identity.imags[i * identity.width + j], 0.0, 0.00001);
                } else {
                    assert_float_closeness(identity.reals[i * identity.width + j], 0.0, 0.00001);
                    assert_float_closeness(identity.imags[i * identity.width + j], 0.0, 0.00001);
                }
            }
        }
    }
}
