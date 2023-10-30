//! Functionality for measurement
use crate::{
    core::State,
    gates::{apply, Gate},
    math::modulus,
};
use rand_distr::{Binomial, Distribution};

/// Single qubit measurement
pub fn measure_qubit(state: &mut State, target: usize, reset: bool, v: Option<u8>) -> u8 {
    let mut prob0 = 0.0;
    let mut prob1 = 0.0;
    let num_pairs = state.len() >> 1;
    let distance = 1 << target;

    for i in 0..num_pairs {
        let s0 = i + ((i >> target) << target);
        let s1 = s0 + distance;

        prob0 += modulus(state.reals[s0], state.imags[s0]).powi(2);
        prob1 += modulus(state.reals[s1], state.imags[s1]).powi(2);
    }

    let val = if let Some(_v) = v {
        assert!(_v == 0 || _v == 1);
        _v
    } else {
        let bin = Binomial::new(1, prob1).unwrap();
        bin.sample(&mut rand::thread_rng()) as u8
    };

    if val == 0 {
        for i in 0..num_pairs {
            let s0 = i + ((i >> target) << target);
            let s1 = s0 + distance;

            state.reals[s0] /= prob0.sqrt();
            state.imags[s0] /= prob0.sqrt();
            state.reals[s1] = 0.0;
            state.imags[s1] = 0.0;
        }
    } else {
        for i in 0..num_pairs {
            let s0 = i + ((i >> target) << target);
            let s1 = s0 + distance;

            state.reals[s0] = 0.0;
            state.imags[s0] = 0.0;
            state.reals[s1] /= prob1.sqrt();
            state.imags[s1] /= prob1.sqrt();
        }

        if reset {
            apply(Gate::X, state, target);
        }
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{assert_float_closeness, gen_random_state};

    #[test]
    fn test_measure_qubit() {
        let mut state = gen_random_state(3);
        println!("{state}");
        let sum = state
            .reals
            .iter()
            .zip(state.imags.iter())
            .map(|(re, im)| modulus(*re, *im).powi(2))
            .sum();
        assert_float_closeness(sum, 1.0, 0.001);

        measure_qubit(&mut state, 0, true, Some(0));
        println!("{state}");
        let sum = state
            .reals
            .iter()
            .zip(state.imags.iter())
            .map(|(re, im)| modulus(*re, *im).powi(2))
            .sum();
        assert_float_closeness(sum, 1.0, 0.001);

        measure_qubit(&mut state, 1, true, Some(0));
        println!("{state}");
        let sum = state
            .reals
            .iter()
            .zip(state.imags.iter())
            .map(|(re, im)| modulus(*re, *im).powi(2))
            .sum();
        assert_float_closeness(sum, 1.0, 0.001);

        measure_qubit(&mut state, 2, true, Some(1));
        println!("{state}");
        let sum = state
            .reals
            .iter()
            .zip(state.imags.iter())
            .map(|(re, im)| modulus(*re, *im).powi(2))
            .sum();

        assert_float_closeness(sum, 1.0, 0.001);
    }

    #[test]
    fn test_measure_qubit_known_state() {
        let n = 3;
        let mut reals = Vec::with_capacity(1 << n);
        let mut imags = Vec::with_capacity(1 << n);

        let vals = vec![
            0.034172256444052966,
            0.29007027387615136,
            -0.1300556493088507,
            0.47222164829858637,
            -0.032338373524095645,
            0.26511510737291843,
            0.1259630181898572,
            -0.09645897805840803,
            -0.31931099330088214,
            -0.24644972468157703,
            -0.15963222942036193,
            -0.14329373536970438,
            -0.1564141838467382,
            -0.4751067410290973,
            0.1034273381193853,
            -0.32966556091031934,
        ];

        let mut i = 0;
        while i < vals.len() - 1 {
            reals.push(vals[i]);
            imags.push(vals[i + 1]);
            i += 2;
        }

        let mut state = State { reals, imags, n };

        let epsilon = 0.001;

        measure_qubit(&mut state, 0, true, Some(0));

        assert_float_closeness(state.reals[0], 0.04528096797370981, epsilon);
        assert_float_closeness(state.imags[0], 0.38436627101331156, epsilon);
        assert_float_closeness(state.reals[1], 0.0, epsilon);
        assert_float_closeness(state.imags[1], 0.0, epsilon);

        assert_float_closeness(state.reals[2], -0.042850926694402595, epsilon);
        assert_float_closeness(state.imags[2], 0.3512986830692283, epsilon);
        assert_float_closeness(state.reals[3], 0.0, epsilon);
        assert_float_closeness(state.imags[3], 0.0, epsilon);

        assert_float_closeness(state.reals[4], -0.42311255872092046, epsilon);
        assert_float_closeness(state.imags[4], -0.32656556082875193, epsilon);
        assert_float_closeness(state.reals[5], 0.0, epsilon);
        assert_float_closeness(state.imags[5], 0.0, epsilon);

        assert_float_closeness(state.reals[6], -0.2072612811212442, epsilon);
        assert_float_closeness(state.imags[6], -0.6295543626114914, epsilon);
        assert_float_closeness(state.reals[7], 0.0, epsilon);
        assert_float_closeness(state.imags[7], 0.0, epsilon);
        println!("{state}");

        measure_qubit(&mut state, 1, true, Some(0));
        assert_float_closeness(state.reals[0], 0.06861878352538178, epsilon);
        assert_float_closeness(state.imags[0], 0.5824686866330654, epsilon);
        assert_float_closeness(state.reals[1], 0.0, epsilon);
        assert_float_closeness(state.imags[1], 0.0, epsilon);

        assert_float_closeness(state.reals[2], 0.0, epsilon);
        assert_float_closeness(state.imags[2], 0.0, epsilon);
        assert_float_closeness(state.reals[3], 0.0, epsilon);
        assert_float_closeness(state.imags[3], 0.0, epsilon);

        assert_float_closeness(state.reals[4], -0.6411848150109799, epsilon);
        assert_float_closeness(state.imags[4], -0.49487748447346463, epsilon);
        assert_float_closeness(state.reals[5], 0.0, epsilon);
        assert_float_closeness(state.imags[5], 0.0, epsilon);

        assert_float_closeness(state.reals[6], 0.0, epsilon);
        assert_float_closeness(state.imags[6], 0.0, epsilon);
        assert_float_closeness(state.reals[7], 0.0, epsilon);
        assert_float_closeness(state.imags[7], 0.0, epsilon);
        println!("{state}");

        measure_qubit(&mut state, 2, true, Some(1));
        assert_float_closeness(state.reals[0], -0.7916334352111761, epsilon);
        assert_float_closeness(state.imags[0], -0.6109963209838112, epsilon);
        assert_float_closeness(state.reals[1], 0.0, epsilon);
        assert_float_closeness(state.imags[1], 0.0, epsilon);

        assert_float_closeness(state.reals[2], 0.0, epsilon);
        assert_float_closeness(state.imags[2], 0.0, epsilon);
        assert_float_closeness(state.reals[3], 0.0, epsilon);
        assert_float_closeness(state.imags[3], 0.0, epsilon);

        assert_float_closeness(state.reals[4], 0.0, epsilon);
        assert_float_closeness(state.imags[4], 0.0, epsilon);
        assert_float_closeness(state.reals[5], 0.0, epsilon);
        assert_float_closeness(state.imags[5], 0.0, epsilon);

        assert_float_closeness(state.reals[6], 0.0, epsilon);
        assert_float_closeness(state.imags[6], 0.0, epsilon);
        assert_float_closeness(state.reals[7], 0.0, epsilon);
        assert_float_closeness(state.imags[7], 0.0, epsilon);
        println!("{state}");
    }
}
