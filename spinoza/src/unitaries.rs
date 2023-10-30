//! Functionality for applying large 2^n * 2^n matrices to the state
//! Ideally, this should be a last resort
use crate::{
    core::State,
    gates::Gate,
    math::{Amplitude, Float, SQRT_ONE_HALF},
};

/// A representation of a Unitary Matrix
pub struct Unitary {
    reals: Vec<Float>,
    imags: Vec<Float>,
    /// The number of rows in the matrix
    height: usize,
    /// The number of columns in the matrix
    width: usize,
}

impl Unitary {
    /// Construct a Unitary from a single qubit gate
    pub fn from_single_qubit_gate(state: &State, gate: Gate, target: usize) -> Self {
        let g = match gate {
            Gate::H => [
                Amplitude {
                    re: SQRT_ONE_HALF,
                    im: 0.0,
                },
                Amplitude {
                    re: SQRT_ONE_HALF,
                    im: 0.0,
                },
                Amplitude {
                    re: SQRT_ONE_HALF,
                    im: 0.0,
                },
                Amplitude {
                    re: -SQRT_ONE_HALF,
                    im: 0.0,
                },
            ],
            Gate::X => [
                Amplitude { re: 0.0, im: 0.0 },
                Amplitude { re: 1.0, im: 0.0 },
                Amplitude { re: 1.0, im: 0.0 },
                Amplitude { re: 0.0, im: 0.0 },
            ],
            Gate::Y => [
                Amplitude { re: 0.0, im: 0.0 },
                Amplitude { re: 0.0, im: -1.0 },
                Amplitude { re: 0.0, im: 1.0 },
                Amplitude { re: 0.0, im: 0.0 },
            ],

            Gate::Z => [
                Amplitude { re: 1.0, im: 0.0 },
                Amplitude { re: 0.0, im: 0.0 },
                Amplitude { re: 0.0, im: 0.0 },
                Amplitude { re: -1.0, im: 0.0 },
            ],
            Gate::P(theta) => [
                Amplitude { re: 1.0, im: 0.0 },
                Amplitude { re: 0.0, im: 0.0 },
                Amplitude { re: 0.0, im: 0.0 },
                Amplitude {
                    re: theta.cos(),
                    im: theta.sin(),
                },
            ],
            Gate::RX(theta) => {
                let theta = theta / 2.0;
                [
                    Amplitude {
                        re: theta.cos(),
                        im: 0.0,
                    },
                    Amplitude {
                        re: 0.0,
                        im: -theta.sin(),
                    },
                    Amplitude {
                        re: 0.0,
                        im: -theta.sin(),
                    },
                    Amplitude {
                        re: theta.cos(),
                        im: 0.0,
                    },
                ]
            }
            Gate::RY(theta) => {
                let theta = theta / 2.0;
                [
                    Amplitude {
                        re: theta.cos(),
                        im: 0.0,
                    },
                    Amplitude {
                        re: -theta.sin(),
                        im: 0.0,
                    },
                    Amplitude {
                        re: theta.sin(),
                        im: 0.0,
                    },
                    Amplitude {
                        re: theta.cos(),
                        im: 0.0,
                    },
                ]
            }
            Gate::RZ(theta) => {
                let theta = theta / 2.0;
                [
                    Amplitude {
                        re: theta.cos(),
                        im: -theta.sin(),
                    },
                    Amplitude { re: 0.0, im: 0.0 },
                    Amplitude { re: 0.0, im: 0.0 },
                    Amplitude {
                        re: theta.cos(),
                        im: theta.sin(),
                    },
                ]
            }
            Gate::U((theta, phi, lambda)) => {
                let theta = theta / 2.0;
                [
                    Amplitude {
                        re: theta.cos(),
                        im: 0.0,
                    },
                    Amplitude {
                        re: -phi.cos() * theta.sin(),
                        im: lambda.sin() * theta.sin(),
                    },
                    Amplitude {
                        re: phi.cos() * theta.sin(),
                        im: phi.sin() * theta.sin(),
                    },
                    Amplitude {
                        re: (phi + lambda).cos() * theta.cos(),
                        im: (phi + lambda).sin() * theta.cos(),
                    },
                ]
            }
            _ => unimplemented!(),
        };
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
    use crate::gates::{apply, Gate};

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
        let u = Unitary::from_single_qubit_gate(&state, Gate::U((1.0, 2.0, 3.0)), 0);

        let s = apply_unitary(&state, &u);

        let mut s1 = State::new(n);
        apply(Gate::U((1.0, 2.0, 3.0)), &mut s1, 0);

        assert_eq!(s.n, s1.n);
        assert_eq!(s.reals, s1.reals);
        assert_eq!(s.imags, s1.imags);
    }
}
