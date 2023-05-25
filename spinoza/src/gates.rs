use crate::{
    core::State,
    math::{Amplitude, Float, SQRT_ONE_HALF},
};
use std::ops::Range;

#[derive(Clone, Copy)]
pub enum Gate {
    H,
    X,
    Y,
    Z,
    P(Float),
    RX(Float),
    RY(Float),
    RZ(Float),
    U((Float, Float, Float)),
}

pub fn apply(gate: Gate, state: &mut State, target: usize) {
    match gate {
        Gate::H => h_apply(state, target),
        Gate::X => x_apply(state, target),
        Gate::Y => y_apply(state, target),
        Gate::Z => z_apply(state, target),
        Gate::P(theta) => p_apply(state, target, theta),
        Gate::RX(theta) => rx_apply(state, target, theta),
        Gate::RY(theta) => ry_apply(state, target, theta),
        Gate::RZ(theta) => rz_apply(state, target, theta),
        Gate::U((theta, phi, lambda)) => u_apply(state, target, theta, phi, lambda),
    }
}

pub fn c_apply(gate: Gate, state: &mut State, control: usize, target: usize) {
    match gate {
        Gate::X => x_c_apply(state, control, target),
        Gate::Y => y_c_apply(state, control, target),
        Gate::P(theta) => p_c_apply(state, control, target, theta),
        Gate::RX(theta) => rx_c_apply(state, control, target, theta),
        Gate::RY(theta) => ry_c_apply(state, control, target, theta),
        _ => todo!(),
    }
}

pub fn cc_apply(gate: Gate, state: &mut State, control0: usize, control1: usize, target: usize) {
    match gate {
        Gate::X => x_cc_apply(state, control0, control1, target),
        _ => todo!(),
    }
}

fn x_apply_target_0(state: &mut State, l0: usize) {
    state.reals.swap(l0, l0 + 1);
    state.imags.swap(l0, l0 + 1);
}

fn x_apply_target(state: &mut State, l0: usize, l1: usize) {
    state.reals.swap(l0, l1);
    state.imags.swap(l0, l1);
}

fn x_proc_chunk(state: &mut State, chunk: usize, target: usize) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let l0 = base + i;
        let l1 = l0 + dist;
        x_apply_target(state, l0, l1)
    }
}

fn x_apply(state: &mut State, target: usize) {
    if target == 0 {
        (0..state.len()).step_by(2).for_each(|l0| {
            x_apply_target_0(state, l0);
        });
    } else {
        let end = state.len() >> 1;
        let chunks = end >> target;
        (0..chunks).for_each(|chunk| {
            x_proc_chunk(state, chunk, target);
        });
    }
}

fn x_c_apply_to_range(state: &mut State, range: Range<usize>, control: usize, target: usize) {
    let dist = 1 << target;
    let marks = (target.min(control), target.max(control));

    for i in range {
        let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
        let l1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
        let l0 = l1 - dist;
        unsafe {
            state.reals.swap_unchecked(l0, l1);
            state.imags.swap_unchecked(l0, l1);
        }
    }
}

fn x_c_apply(state: &mut State, control: usize, target: usize) {
    let end = state.len() >> 2;
    x_c_apply_to_range(state, 0..end, control, target);
}

pub fn x_cc_apply(state: &mut State, control0: usize, control1: usize, target: usize) {
    let mut i = 0;
    let dist = 1 << target;

    while i < state.len() {
        let c0c1_set = (((i >> control0) & 1) != 0) && (((i >> control1) & 1) != 0);
        if c0c1_set {
            let l0 = i;
            let l1 = i + dist;
            unsafe {
                let temp0 = *state.reals.get_unchecked(l0);
                *state.reals.get_unchecked_mut(l0) = *state.reals.get_unchecked(l1);
                *state.reals.get_unchecked_mut(l1) = temp0;

                let temp1 = *state.imags.get_unchecked(l0);
                *state.imags.get_unchecked_mut(l0) = *state.imags.get_unchecked(l1);
                *state.imags.get_unchecked_mut(l1) = temp1;
            };
            i += dist;
        }
        i += 1;
    }
}

fn y_apply_target(state: &mut State, l0: usize, l1: usize) {
    state.reals.swap(l0, l1);
    state.imags.swap(l0, l1);

    std::mem::swap(&mut state.reals[l0], &mut state.imags[l0]);
    std::mem::swap(&mut state.reals[l1], &mut state.imags[l1]);
    state.imags[l0] = -state.imags[l0];
    state.reals[l1] = -state.reals[l1];
}

fn y_proc_chunk(state: &mut State, chunk: usize, target: usize) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let l0 = base + i;
        let l1 = l0 + dist;
        y_apply_target(state, l0, l1);
    }
}

fn y_apply(state: &mut State, target: usize) {
    let end = state.len() >> 1;
    let chunks = end >> target;

    (0..chunks).for_each(|chunk| {
        y_proc_chunk(state, chunk, target);
    });
}

fn y_c_apply_to_range(state: &mut State, range: Range<usize>, control: usize, target: usize) {
    let dist = 1 << target;
    let marks = (target.min(control), target.max(control));

    for i in range {
        let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
        let l1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
        let l0 = l1 - dist;
        y_apply_target(state, l0, l1);
    }
}

fn y_c_apply(state: &mut State, control: usize, target: usize) {
    let end = state.len() >> 2;
    y_c_apply_to_range(state, 0..end, control, target);
}

fn h_apply_target_0(state: &mut State, l0: usize) {
    let a = state.reals[l0];
    let b = state.imags[l0];

    let c = state.reals[l0 + 1];
    let d = state.imags[l0 + 1];

    let a1 = SQRT_ONE_HALF * a;
    let b1 = SQRT_ONE_HALF * b;
    let c1 = SQRT_ONE_HALF * c;
    let d1 = SQRT_ONE_HALF * d;

    state.reals[l0] = a1 + c1;
    state.imags[l0] = b1 + d1;
    state.reals[l0 + 1] = a1 - c1;
    state.imags[l0 + 1] = b1 - d1;
}

fn h_apply_target(state: &mut State, l0: usize, l1: usize) {
    let a = state.reals[l0];
    let b = state.imags[l0];
    let c = state.reals[l1];
    let d = state.imags[l1];

    let a1 = SQRT_ONE_HALF * a;
    let b1 = SQRT_ONE_HALF * b;
    let c1 = SQRT_ONE_HALF * c;
    let d1 = SQRT_ONE_HALF * d;

    state.reals[l0] = a1 + c1;
    state.imags[l0] = b1 + d1;
    state.reals[l1] = a1 - c1;
    state.imags[l1] = b1 - d1;
}

fn h_proc_chunk(state: &mut State, chunk: usize, target: usize) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let l0 = base + i;
        let l1 = l0 + dist;
        h_apply_target(state, l0, l1);
    }
}

fn h_apply(state: &mut State, target: usize) {
    if target == 0 {
        (0..state.len())
            .step_by(2)
            .for_each(|l| h_apply_target_0(state, l));
    } else {
        let end = state.len() >> 1;
        let chunks = end >> target;
        (0..chunks).for_each(|c| {
            h_proc_chunk(state, c, target);
        });
    }
}

fn rx_apply_target_0(state: &mut State, l: usize, cos: Float, neg_sin: Float) {
    unsafe {
        let a = *state.reals.get_unchecked(l);
        let b = *state.imags.get_unchecked(l);

        // c + id
        let c = *state.reals.get_unchecked(l + 1);
        let d = *state.imags.get_unchecked(l + 1);

        *state.reals.get_unchecked_mut(l) = a.mul_add(cos, d * -neg_sin);
        *state.imags.get_unchecked_mut(l) = b.mul_add(cos, c * neg_sin);

        *state.reals.get_unchecked_mut(l + 1) = b.mul_add(-neg_sin, c * cos);
        *state.imags.get_unchecked_mut(l + 1) = d.mul_add(cos, a * neg_sin);
    }
}

fn rx_apply_target(state: &mut State, l0: usize, l1: usize, cos: Float, neg_sin: Float) {
    unsafe {
        // a + ib
        let a = *state.reals.get_unchecked(l0);
        let b = *state.imags.get_unchecked(l0);

        // c + id
        let c = *state.reals.get_unchecked(l1);
        let d = *state.imags.get_unchecked(l1);

        *state.reals.get_unchecked_mut(l0) = a.mul_add(cos, d * -neg_sin);
        *state.imags.get_unchecked_mut(l0) = b.mul_add(cos, c * neg_sin);

        *state.reals.get_unchecked_mut(l1) = b.mul_add(-neg_sin, c * cos);
        *state.imags.get_unchecked_mut(l1) = d.mul_add(cos, a * neg_sin);
    }
}

pub fn rx_proc_chunk(state: &mut State, chunk: usize, target: usize, cos: Float, neg_sin: Float) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let l0 = base + i;
        let l1 = l0 + dist;
        rx_apply_target(state, l0, l1, cos, neg_sin);
    }
}

pub fn rx_apply(state: &mut State, target: usize, angle: Float) {
    let theta = angle * 0.5;
    let ct = Float::cos(theta);
    let nst = -Float::sin(theta);

    if target == 0 {
        let mut l = 0;
        while l < state.len() - 1 {
            rx_apply_target_0(state, l, ct, nst);
            l += 2;
        }
    } else {
        let end = state.len() >> 1;
        let chunks = end >> target;
        (0..chunks).for_each(|chunk| {
            rx_proc_chunk(state, chunk, target, ct, nst);
        });
    }
}

fn rx_c_apply_to_range(
    state: &mut State,
    range: Range<usize>,
    control: usize,
    target: usize,
    cos: Float,
    neg_sin: Float,
) {
    let dist = 1 << target;
    let marks = (target.min(control), target.max(control));

    for i in range {
        let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
        let l1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
        let l0 = l1 - dist;
        rx_apply_target(state, l0, l1, cos, neg_sin);
    }
}

fn rx_c_apply(state: &mut State, control: usize, target: usize, angle: Float) {
    let theta = angle * 0.5;
    let ct = Float::cos(theta);
    let nst = -Float::sin(theta);
    let end = state.len() >> 2;
    rx_c_apply_to_range(state, 0..end, control, target, ct, nst);
}

fn p_apply_target(state: &mut State, l1: usize, cos: Float, sin: Float) {
    let z_re = state.reals[l1];
    let z_im = state.imags[l1];
    state.reals[l1] = z_re.mul_add(cos, -z_im * sin);
    state.imags[l1] = z_im.mul_add(cos, z_re * sin);
}

fn p_proc_chunk(state: &mut State, chunk: usize, target: usize, cos: Float, sin: Float) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;

    // l1 = base + i + dist, for i \in \{0, 1, 2, \ldots, dist-1\}
    for l1 in (base + dist)..(base + 2 * dist) {
        p_apply_target(state, l1, cos, sin);
    }
}

fn p_apply(state: &mut State, target: usize, angle: Float) {
    let (sin, cos) = Float::sin_cos(angle);
    let end = state.len() >> 1;
    let chunks = end >> target;

    (0..chunks).for_each(|chunk| {
        p_proc_chunk(state, chunk, target, cos, sin);
    });
}

fn p_c_apply_to_range(
    state: &mut State,
    range: Range<usize>,
    control: usize,
    target: usize,
    cos: Float,
    sin: Float,
) {
    let marks = (target.min(control), target.max(control));

    for i in range {
        let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
        let l1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
        p_apply_target(state, l1, cos, sin);
    }
}

fn p_c_apply(state: &mut State, control: usize, target: usize, angle: Float) {
    let (sin, cos) = Float::sin_cos(angle);
    let end = state.len() >> 2;
    p_c_apply_to_range(state, 0..end, control, target, cos, sin);
}

fn rz_apply_target(state: &mut State, i: usize, m: &Amplitude) {
    unsafe {
        // a + ib
        let a = *state.reals.get_unchecked(i);
        let b = *state.imags.get_unchecked(i);

        *state.reals.get_unchecked_mut(i) = a.mul_add(m.re, -b * m.im);
        *state.imags.get_unchecked_mut(i) = a.mul_add(m.im, b * m.re);
    }
}

fn rz_apply_strategy1(state: &mut State, target: usize, diag_matrix: &[Amplitude; 2]) {
    let mut chunk_start = 0;

    while chunk_start < state.len() {
        let chunk_end = state.len().min(((chunk_start >> target) + 1) << target);

        let m = unsafe { diag_matrix.get_unchecked((chunk_start >> target) & 1) };
        for i in chunk_start..chunk_end {
            rz_apply_target(state, i, m);
        }
        chunk_start = chunk_end;
    }
}

// NOTE: since we are checking pairs, rather than generating, we need to go through the *entire*
// state
fn rz_apply(state: &mut State, target: usize, angle: Float) {
    let theta = angle * 0.5;
    let (s, c) = Float::sin_cos(theta);
    let d0 = Amplitude { re: c, im: -s };
    let d1 = Amplitude { re: c, im: s };
    let diag_matrix = [d0, d1];
    rz_apply_strategy1(state, target, &diag_matrix);
}

fn z_proc_chunk(state: &mut State, chunk: usize, target: usize) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in base + dist..base + 2 * dist {
        unsafe {
            *state.reals.get_unchecked_mut(i) = -state.reals.get_unchecked(i);
            *state.imags.get_unchecked_mut(i) = -state.imags.get_unchecked(i);
        };
    }
}

fn z_apply(state: &mut State, target: usize) {
    // NOTE: chunks == end >> target, where end == state.len() >> 1
    let chunks = state.len() >> (target + 1);
    (0..chunks).for_each(|c| z_proc_chunk(state, c, target));
}

#[inline]
fn ry_apply_target(state: &mut State, l0: usize, l1: usize, sin: Float, cos: Float) {
    // a + ib
    let a = state.reals[l0];
    let b = state.imags[l0];

    // c + id
    let c = state.reals[l1];
    let d = state.imags[l1];

    state.reals[l0] = a * cos - c * sin;
    state.imags[l0] = b * cos - d * sin;
    state.reals[l1] = a * sin + c * cos;
    state.imags[l1] = b * sin + d * cos;
}

fn ry_apply_strategy2(state: &mut State, chunk: usize, target: usize, sin: Float, cos: Float) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let l0 = base + i;
        let l1 = l0 + dist;
        ry_apply_target(state, l0, l1, sin, cos);
    }
}

fn ry_apply(state: &mut State, target: usize, angle: Float) {
    let theta = angle * 0.5;
    let (sin, cos) = Float::sin_cos(theta);

    let end = state.len() >> 1;
    let chunks = end >> target;

    (0..chunks).for_each(|chunk| {
        ry_apply_strategy2(state, chunk, target, sin, cos);
    });
}

fn ry_c_apply_strategy3(
    state: &mut State,
    range: Range<usize>,
    control: usize,
    target: usize,
    angle: Float,
) {
    let theta = angle * 0.5;
    let (sin, cos) = Float::sin_cos(theta);
    let dist = 1 << target;
    let marks = (target.min(control), target.max(control));

    for i in range {
        let x = i + (1 << (marks.1 - 1)) + ((i >> (marks.1 - 1)) << (marks.1 - 1));
        let l1 = x + (1 << marks.0) + ((x >> marks.0) << marks.0);
        let l0 = l1 - dist;
        ry_apply_target(state, l0, l1, sin, cos);
    }
}

fn ry_c_apply(state: &mut State, control: usize, target: usize, angle: Float) {
    let end = state.len() >> 2;
    ry_c_apply_strategy3(state, 0..end, control, target, angle);
}

fn u_apply(state: &mut State, target: usize, theta: Float, phi: Float, lambda: Float) {
    let c = Amplitude {
        re: Float::cos(theta * 0.5),
        im: 0.0,
    };
    let ncs = Amplitude {
        re: -Float::cos(lambda) * Float::sin(theta * 0.5),
        im: -Float::sin(lambda) * Float::sin(theta * 0.5),
    };
    let es = Amplitude {
        re: Float::cos(phi) * Float::sin(theta * 0.5),
        im: Float::sin(lambda) * Float::sin(theta * 0.5),
    };
    let ec = Amplitude {
        re: Float::cos(phi + lambda) * Float::cos(theta * 0.5),
        im: Float::sin(phi + lambda) * Float::cos(theta * 0.5),
    };
    let g = [c, ncs, es, ec];

    // NOTE: chunks == end >> target, where end == state.len() >> 1
    let chunks = state.len() >> (target + 1);

    (0..chunks).for_each(|chunk| {
        u_apply_strategy2(state, &g, chunk, target);
    });
}

fn u_apply_target(state: &mut State, g: &[Amplitude], l0: usize, l1: usize) {
    let z0_re = state.reals[l0];
    let z0_im = state.imags[l0];
    let z1_re = state.reals[l1];
    let z1_im = state.imags[l1];

    let c = z0_re;
    let d = z0_im;
    let m = z1_re;
    let n = z1_im;

    let a = g[0].re;
    let b = g[0].im;
    let k = g[1].re;
    let l = g[1].im;

    let q = g[2].re;
    let r = g[2].im;
    let s = g[3].re;
    let t = g[3].im;

    state.reals[l0] = a.mul_add(c, (-b).mul_add(d, k.mul_add(m, -l * n)));
    state.imags[l0] = a.mul_add(d, b.mul_add(c, k.mul_add(n, l * m)));
    state.reals[l1] = q.mul_add(c, (-r).mul_add(d, s.mul_add(m, -t * n)));
    state.imags[l1] = q.mul_add(d, r.mul_add(c, s.mul_add(n, t * m)));
}

pub fn u_apply_strategy2(state: &mut State, g: &[Amplitude], chunk: usize, target: usize) {
    let dist = 1 << target;
    let base = (2 * chunk) << target;
    for i in 0..dist {
        let l0 = base + i;
        let l1 = l0 + dist;
        u_apply_target(state, g, l0, l1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::iqft,
        math::{pow2f, PI},
        utils::assert_float_closeness,
    };

    fn qcbm_functional(n: usize) -> State {
        let mut state = State::new(n);
        let pairs: Vec<_> = (0..n).map(|i| (i, (i + 1) % n)).collect();

        for i in 0..n {
            apply(Gate::RX(1.0), &mut state, i);
            apply(Gate::RZ(1.0), &mut state, i);
        }

        for i in 0..n - 1 {
            let (p0, p1) = pairs[i];
            c_apply(Gate::X, &mut state, p0, p1);
        }

        for _ in 0..9 {
            for i in 0..n {
                apply(Gate::RZ(1.0), &mut state, i);
                apply(Gate::RX(1.0), &mut state, i);
                apply(Gate::RZ(1.0), &mut state, i);
            }

            for i in 0..n - 1 {
                let (p0, p1) = pairs[i];
                c_apply(Gate::X, &mut state, p0, p1);
            }
        }

        for i in 0..n {
            apply(Gate::RZ(1.0), &mut state, i);
            apply(Gate::RX(1.0), &mut state, i);
        }
        state
    }

    #[test]
    fn h_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::H, &mut state, i);
        }

        for i in 0..(1 << n) {
            assert_float_closeness(state.reals[i], 0.35355339059327384, 1e-10);
            assert_float_closeness(state.imags[i], 0.0, 1e-10);
        }
    }

    #[test]
    fn x_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::X, &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.0, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[7], 1.0, 1e-10);
        assert_float_closeness(state.imags[7], 0.0, 1e-10);
    }

    #[test]
    fn x_gate_20_qubits() {
        let n = 20;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::X, &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.0, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[(1 << n) - 1], 1.0, 1e-10);
        assert_float_closeness(state.imags[(1 << n) - 1], 0.0, 1e-10);
    }

    #[test]
    fn y_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::Y, &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.0, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[7], 0.0, 1e-10);
        assert_float_closeness(state.imags[7], -1.0, 1e-10);
    }

    #[test]
    fn y_gate_20_qubits() {
        let n = 20;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::Y, &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.0, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[(1 << n) - 1], 1.0, 1e-10);
        assert_float_closeness(state.imags[(1 << n) - 1], 0.0, 1e-10);
    }

    #[test]
    fn z_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::Z, &mut state, i);
        }

        for i in 0..(1 << n) {
            if i == 0 {
                assert_float_closeness(state.reals[i], 1.0, 1e-10);
                assert_float_closeness(state.imags[i], 0.0, 1e-10);
            } else {
                assert_float_closeness(state.reals[i], 0.0, 1e-10);
                assert_float_closeness(state.imags[i], 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn p_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::P(PI), &mut state, i);
        }

        for i in 0..(1 << n) {
            if i == 0 {
                assert_float_closeness(state.reals[i], 1.0, 1e-10);
                assert_float_closeness(state.imags[i], 0.0, 1e-10);
            } else {
                assert_float_closeness(state.reals[i], 0.0, 1e-10);
                assert_float_closeness(state.imags[i], 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn rx_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::RX(1.0), &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.6758712218347053, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[1], 0.0, 1e-10);
        assert_float_closeness(state.imags[1], -0.3692301313020644, 1e-10);

        assert_float_closeness(state.reals[2], 0.0, 1e-10);
        assert_float_closeness(state.imags[2], -0.3692301313020644, 1e-10);

        assert_float_closeness(state.reals[3], -0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[3], 0.0, 1e-10);

        assert_float_closeness(state.reals[4], 0.0, 1e-10);
        assert_float_closeness(state.imags[4], -0.3692301313020644, 1e-10);

        assert_float_closeness(state.reals[5], -0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[5], 0.0, 1e-10);

        assert_float_closeness(state.reals[6], -0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[6], 0.0, 1e-10);

        assert_float_closeness(state.reals[7], 0.0, 1e-10);
        assert_float_closeness(state.imags[7], 0.11019540730213864, 1e-10);
    }

    #[test]
    fn ry_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::RY(1.0), &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.6758712218347053, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[1], 0.3692301313020644, 1e-10);
        assert_float_closeness(state.imags[1], 0.0, 1e-10);

        assert_float_closeness(state.reals[2], 0.3692301313020644, 1e-10);
        assert_float_closeness(state.imags[2], 0.0, 1e-10);

        assert_float_closeness(state.reals[3], 0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[3], 0.0, 1e-10);

        assert_float_closeness(state.reals[4], 0.3692301313020644, 1e-10);
        assert_float_closeness(state.imags[4], 0.0, 1e-10);

        assert_float_closeness(state.reals[5], 0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[5], 0.0, 1e-10);

        assert_float_closeness(state.reals[6], 0.20171134005566746, 1e-10);
        assert_float_closeness(state.imags[6], 0.0, 1e-10);

        assert_float_closeness(state.reals[7], 0.11019540730213864, 1e-10);
        assert_float_closeness(state.imags[7], 0.0, 1e-10);
    }

    #[test]
    fn rz_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::RZ(1.0), &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.07073720166770296, 1e-10);
        assert_float_closeness(state.imags[0], -0.9974949866040546, 1e-10);

        assert_float_closeness(state.reals[1], 0.0, 1e-10);
        assert_float_closeness(state.imags[1], 0.0, 1e-10);

        assert_float_closeness(state.reals[2], 0.0, 1e-10);
        assert_float_closeness(state.imags[2], 0.0, 1e-10);

        assert_float_closeness(state.reals[3], 0.0, 1e-10);
        assert_float_closeness(state.imags[3], 0.0, 1e-10);

        assert_float_closeness(state.reals[4], 0.0, 1e-10);
        assert_float_closeness(state.imags[4], 0.0, 1e-10);

        assert_float_closeness(state.reals[5], 0.0, 1e-10);
        assert_float_closeness(state.imags[5], 0.0, 1e-10);

        assert_float_closeness(state.reals[6], 0.0, 1e-10);
        assert_float_closeness(state.imags[6], 0.0, 1e-10);

        assert_float_closeness(state.reals[7], 0.0, 1e-10);
        assert_float_closeness(state.imags[7], 0.0, 1e-10);
    }

    #[test]
    fn value_encoding_3_qubits() {
        let v = 2.4;
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::H, &mut state, i);
        }
        for i in 0..n {
            apply(Gate::P(2.0 * PI / (pow2f(i + 1)) * v), &mut state, i);
        }
        let targets: Vec<usize> = (0..n).rev().collect();

        iqft(&mut state, &targets);
    }

    #[test]
    fn value_encoding_20_qubits() {
        let v = 2.4;
        let n = 20;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::H, &mut state, i);
        }
        for i in 0..n {
            apply(Gate::P(2.0 * PI / (pow2f(i + 1)) * v), &mut state, i);
        }
        let targets: Vec<usize> = (0..n).rev().collect();

        iqft(&mut state, &targets);
    }

    #[test]
    fn qcbm_3_qubits() {
        let state = qcbm_functional(3);

        assert_float_closeness(state.reals[0], 0.18037770683997864, 1e-10);
        assert_float_closeness(state.imags[0], -0.17626993141958947, 1e-10);

        assert_float_closeness(state.reals[7], 0.014503954556966365, 1e-10);
        assert_float_closeness(state.imags[7], -0.11198008105074927, 1e-10);
    }

    #[test]
    fn qcbm_20_qubits() {
        let state = qcbm_functional(20);

        assert_float_closeness(state.reals[0], -0.0022221321676945643, 1e-10);
        assert_float_closeness(state.imags[0], 0.001743068112560825, 1e-10);

        assert_float_closeness(state.reals[7], -0.0031017461877124453, 1e-10);
        assert_float_closeness(state.imags[7], -0.0034043237120339686, 1e-10);

        assert_float_closeness(state.reals[12], 0.0005494086136357235, 1e-10);
        assert_float_closeness(state.imags[12], -0.00009827749580581964, 1e-10);
    }

    #[test]
    fn u_gate_3_qubits() {
        let n = 3;
        let mut state = State::new(n);

        for i in 0..n {
            apply(Gate::U((1.0, 1.0, 1.0)), &mut state, i);
        }

        assert_float_closeness(state.reals[0], 0.6758712218347053, 1e-10);
        assert_float_closeness(state.imags[0], 0.0, 1e-10);

        assert_float_closeness(state.reals[1], 0.19949589133850137, 1e-10);
        assert_float_closeness(state.imags[1], 0.3106964422074971, 1e-10);

        assert_float_closeness(state.reals[2], 0.19949589133850137, 1e-10);
        assert_float_closeness(state.imags[2], 0.3106964422074971, 1e-10);

        assert_float_closeness(state.reals[3], -0.08394153605985091, 1e-10);
        assert_float_closeness(state.imags[3], 0.18341560247417849, 1e-10);

        assert_float_closeness(state.reals[2], 0.19949589133850137, 1e-10);
        assert_float_closeness(state.imags[2], 0.3106964422074971, 1e-10);

        assert_float_closeness(state.reals[3], -0.08394153605985091, 1e-10);
        assert_float_closeness(state.imags[3], 0.18341560247417849, 1e-10);

        assert_float_closeness(state.reals[3], -0.08394153605985091, 1e-10);
        assert_float_closeness(state.imags[3], 0.18341560247417849, 1e-10);

        assert_float_closeness(state.reals[7], -0.1090926263889472, 1e-10);
        assert_float_closeness(state.imags[7], 0.015550776766638148, 1e-10);
    }
}
