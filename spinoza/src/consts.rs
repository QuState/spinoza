//! Consts for Quantum State Simulation, such as the Pauli Gates and the Hadamard gate
use crate::math::{Amplitude, SQRT_ONE_HALF};

/// The 2 x 2 matrix representation of the Hadamard gate
pub const H: [Amplitude; 4] = [
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
];

/// The 2 x 2 matrix representation of the X gate
pub const X: [Amplitude; 4] = [
    Amplitude { re: 0.0, im: 0.0 },
    Amplitude { re: 1.0, im: 0.0 },
    Amplitude { re: 1.0, im: 0.0 },
    Amplitude { re: 0.0, im: 0.0 },
];

/// The 2 x 2 matrix representation of the Y gate
pub const Y: [Amplitude; 4] = [
    Amplitude { re: 0.0, im: 0.0 },
    Amplitude { re: 0.0, im: -1.0 },
    Amplitude { re: 0.0, im: 1.0 },
    Amplitude { re: 0.0, im: 0.0 },
];

/// The 2 x 2 matrix representation of the Z gate
pub const Z: [Amplitude; 4] = [
    Amplitude { re: 1.0, im: 0.0 },
    Amplitude { re: 0.0, im: 0.0 },
    Amplitude { re: 0.0, im: 0.0 },
    Amplitude { re: -1.0, im: 0.0 },
];
