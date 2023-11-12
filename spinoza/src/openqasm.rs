//! Functionality for creating a QuantumCircuit from an OpenQASM 2.0 program
use crate::{
    circuit::{QuantumCircuit, QuantumRegister},
    math::{Float, PI},
};
use evalexpr::*;
use qasm::{lex, parse, process, Argument, AstNode};
use std::{collections::HashMap, env, fs::File, io::prelude::*, path::Path};

/// Parse an OpenQASM 2.0 program file, and convert it into a `QuantumCircuit`.
pub fn load(filename: &Path) -> QuantumCircuit {
    let cwd = env::current_dir().unwrap();
    let mut source = String::new();
    let mut f = File::open(filename).expect("cannot find source file");
    f.read_to_string(&mut source).expect("couldn't read file");

    let processed_source = process(&source, &cwd);
    let mut tokens = lex(&processed_source);
    let ast = parse(&mut tokens).unwrap();
    build_circuit(&ast)
}

/// Parse an OpenQASM 2.0 program in `str` format, and convert it into a `QuantumCircuit`.
pub fn loads(qasm_as_str: &str) -> QuantumCircuit {
    let cwd = env::current_dir().unwrap();
    let processed_source = process(qasm_as_str, &cwd);
    let mut tokens = lex(&processed_source);
    let ast = parse(&mut tokens).unwrap();
    build_circuit(&ast)
}

/// Build a circuit from an AST
fn build_circuit(ast: &[AstNode]) -> QuantumCircuit {
    let context = context_map! {
        "pi" => PI,
        "-pi" => -PI,
    }
    .unwrap();

    let mut registers = HashMap::new();

    // Find all registers
    for node in ast.iter() {
        if let AstNode::QReg(identifier, num_qubits) = node {
            let n = (*num_qubits).try_into().unwrap();
            registers.insert(identifier, QuantumRegister::new(n));
        }
    }

    let mut qrs: Vec<&mut QuantumRegister> = registers.iter_mut().map(|(_, r)| r).collect();
    let mut qc = QuantumCircuit::new(&mut qrs);

    for node in ast.iter() {
        match node {
            AstNode::ApplyGate(gate, args0, args1) => match gate.as_str() {
                "h" => {
                    for arg in args0.iter() {
                        if let Argument::Qubit(identifier, qubit_num) = arg {
                            let register = registers.get(identifier).unwrap();
                            qc.h(register[(*qubit_num).try_into().unwrap()]);
                        }
                    }
                }
                "x" => {
                    for arg in args0.iter() {
                        if let Argument::Qubit(identifier, qubit_num) = arg {
                            let register = registers.get(identifier).unwrap();
                            qc.x(register[(*qubit_num).try_into().unwrap()]);
                        }
                    }
                }
                "y" => {
                    for arg in args0.iter() {
                        if let Argument::Qubit(identifier, qubit_num) = arg {
                            let register = registers.get(identifier).unwrap();
                            qc.y(register[(*qubit_num).try_into().unwrap()]);
                        }
                    }
                }
                "z" => {
                    for arg in args0.iter() {
                        if let Argument::Qubit(identifier, qubit_num) = arg {
                            let register = registers.get(identifier).unwrap();
                            qc.z(register[(*qubit_num).try_into().unwrap()]);
                        }
                    }
                }
                "rx" => {
                    for arg in args0.iter() {
                        if let Argument::Qubit(identifier, qubit_num) = arg {
                            let angle = args1[0].replace(' ', "").parse::<Float>().unwrap();
                            let register = registers.get(identifier).unwrap();
                            qc.rx(angle, register[(*qubit_num).try_into().unwrap()]);
                        }
                    }
                }
                "ry" => {
                    for arg in args0.iter() {
                        if let Argument::Qubit(identifier, qubit_num) = arg {
                            let angle = args1[0].replace(' ', "").parse::<Float>().unwrap();
                            let register = registers.get(identifier).unwrap();
                            qc.ry(angle, register[(*qubit_num).try_into().unwrap()]);
                        }
                    }
                }
                "rz" => {
                    for arg in args0.iter() {
                        if let Argument::Qubit(identifier, qubit_num) = arg {
                            let angle = args1[0].replace(' ', "").parse::<Float>().unwrap();
                            let register = registers.get(identifier).unwrap();
                            qc.rz(angle, register[(*qubit_num).try_into().unwrap()]);
                        }
                    }
                }
                "u" => {
                    for arg in args0.iter() {
                        if let Argument::Qubit(identifier, qubit_num) = arg {
                            let theta = args1[0].replace(' ', "").parse::<Float>().unwrap();
                            let phi = args1[1].replace(' ', "").parse::<Float>().unwrap();
                            let lambda = args1[2].replace(' ', "").parse::<Float>().unwrap();
                            let register = registers.get(identifier).unwrap();
                            qc.u(
                                theta,
                                phi,
                                lambda,
                                register[(*qubit_num).try_into().unwrap()],
                            );
                        }
                    }
                }
                "cp" => {
                    let control = if let Argument::Qubit(identifier, qubit_num) = &args0[0] {
                        let register = registers.get(identifier).unwrap();
                        register[(*qubit_num).try_into().unwrap()]
                    } else {
                        panic!("there's no argument 0 for the CP gate")
                    };
                    let target = if let Argument::Qubit(identifier, qubit_num) = &args0[1] {
                        let register = registers.get(identifier).unwrap();
                        register[(*qubit_num).try_into().unwrap()]
                    } else {
                        panic!("there's no argument 1 for the CP gate")
                    };
                    let theta = eval_with_context(&args1[0], &context).unwrap();
                    qc.cp(theta.as_float().unwrap(), control, target);
                }
                "cx" => {
                    let control = if let Argument::Qubit(identifier, qubit_num) = &args0[0] {
                        let register = registers.get(&identifier).unwrap();
                        register[(*qubit_num).try_into().unwrap()]
                    } else {
                        panic!("there's no argument 0 for the CX gate")
                    };
                    let target = if let Argument::Qubit(identifier, qubit_num) = &args0[1] {
                        let register = registers.get(&identifier).unwrap();
                        register[(*qubit_num).try_into().unwrap()]
                    } else {
                        panic!("there's no argument 1 for the CX gate")
                    };
                    qc.cx(control, target);
                }
                _ => todo!(),
            },
            AstNode::QReg(_, _) => (),
            _ => todo!(),
        }
    }
    qc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{circuit::QuantumRegister, utils::assert_float_closeness};

    #[test]
    fn quantum_lstm_load() {
        let filename: &Path = Path::new("../qasm/quantum_lstm.qasm");
        let mut qc1 = load(filename);
        qc1.execute();

        qc1.state
            .reals
            .iter()
            .zip(qc1.state.imags.iter())
            .for_each(|(qc1_re, qc1_im)| {
                println!("{} + i{}", *qc1_re, *qc1_im);
            });
    }

    #[test]
    fn iqft_load() {
        let filename: &Path = Path::new("../qasm/iqft.qasm");
        let mut qc2 = load(filename);
        qc2.execute();

        let n: usize = qc2.state.n.into();
        let targets: Vec<_> = (0..n).map(|n| n).rev().collect();
        let mut qr = QuantumRegister::new(n);
        let mut qc1 = QuantumCircuit::new(&mut [&mut qr]);
        qc1.iqft(&targets);
        qc1.execute();

        qc1.state
            .reals
            .iter()
            .zip(qc1.state.imags.iter())
            .zip(qc2.state.reals.iter())
            .zip(qc2.state.imags.iter())
            .for_each(|(((qc1_re, qc1_im), qc2_re), qc2_im)| {
                assert_float_closeness(*qc1_re, *qc2_re, 0.001);
                assert_float_closeness(*qc1_im, *qc2_im, 0.001);
            });
    }

    #[test]
    fn iqft_loads() {
        let qasm_as_str = include_str!("../../qasm/iqft.qasm");
        let mut qc2 = loads(qasm_as_str);
        qc2.execute();

        let n: usize = qc2.state.n.into();
        let targets: Vec<_> = (0..n).map(|n| n).rev().collect();
        let mut qr = QuantumRegister::new(n);
        let mut qc1 = QuantumCircuit::new(&mut [&mut qr]);
        qc1.iqft(&targets);
        qc1.execute();

        qc1.state
            .reals
            .iter()
            .zip(qc1.state.imags.iter())
            .zip(qc2.state.reals.iter())
            .zip(qc2.state.imags.iter())
            .for_each(|(((qc1_re, qc1_im), qc2_re), qc2_im)| {
                assert_float_closeness(*qc1_re, *qc2_re, 0.001);
                assert_float_closeness(*qc1_im, *qc2_im, 0.001);
            });
    }

    #[test]
    fn non_controlled_gates_load() {
        let filename: &Path = Path::new("../qasm/test0.qasm");
        let mut qc2 = load(filename);
        qc2.execute();

        let n: usize = qc2.state.n.into();
        let mut qr = QuantumRegister::new(n);
        let mut qc1 = QuantumCircuit::new(&mut [&mut qr]);
        qc1.h(0);
        qc1.x(1);
        qc1.y(2);
        qc1.z(3);
        qc1.rx(1.0, 4);
        qc1.ry(2.0, 5);
        qc1.rz(3.0, 6);
        qc1.u(1.0, 2.0, 3.0, 7);
        qc1.execute();

        qc1.state
            .reals
            .iter()
            .zip(qc1.state.imags.iter())
            .zip(qc2.state.reals.iter())
            .zip(qc2.state.imags.iter())
            .for_each(|(((qc1_re, qc1_im), qc2_re), qc2_im)| {
                assert_float_closeness(*qc1_re, *qc2_re, 0.001);
                assert_float_closeness(*qc1_im, *qc2_im, 0.001);
            });
    }

    #[test]
    fn non_controlled_gates_loads() {
        let qasm_as_str = include_str!("../../qasm/test0.qasm");
        let mut qc2 = loads(qasm_as_str);
        qc2.execute();

        let n: usize = qc2.state.n.into();
        let mut qr = QuantumRegister::new(n);
        let mut qc1 = QuantumCircuit::new(&mut [&mut qr]);
        qc1.h(0);
        qc1.x(1);
        qc1.y(2);
        qc1.z(3);
        qc1.rx(1.0, 4);
        qc1.ry(2.0, 5);
        qc1.rz(3.0, 6);
        qc1.u(1.0, 2.0, 3.0, 7);
        qc1.execute();

        qc1.state
            .reals
            .iter()
            .zip(qc1.state.imags.iter())
            .zip(qc2.state.reals.iter())
            .zip(qc2.state.imags.iter())
            .for_each(|(((qc1_re, qc1_im), qc2_re), qc2_im)| {
                assert_float_closeness(*qc1_re, *qc2_re, 0.001);
                assert_float_closeness(*qc1_im, *qc2_im, 0.001);
            });
    }
}
