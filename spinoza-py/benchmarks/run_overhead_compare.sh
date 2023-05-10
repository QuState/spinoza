#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -ne 3 ]]
then
  echo "Usage: $0 <start-qubits> <end-qubits> <iters>"
  exit 1
fi

START_QUBITS=$1
END_QUBITS=$2
ITERS=$3

OUTPUT_DIR=overhead-data.$(date +"%Y.%m.%d.%H-%M-%S")
mkdir -p $OUTPUT_DIR/spinoza && mkdir $OUTPUT_DIR/spinoza-py

gates=("value_encoding")

bm_py_gates () {

    for g in ${gates[@]}; do
        for n in $(seq $START_QUBITS $END_QUBITS); do
            echo "running spinoza-py benchmark for $g with $n qubits" && \
                for i in $(seq 1 $ITERS); do
                    python3 py_${g}.py --qubits ${n} >> ${OUTPUT_DIR}/spinoza-py/${g}-${n}qubits
                done
            done
        done
}

bm_rust_gates () {
    cd ../../spinoza
    cargo clean && cargo update && cargo build --release --examples
    cd ../spinoza-py/benchmarks

    for g in ${gates[@]}; do
        for n in $(seq $START_QUBITS $END_QUBITS); do
            echo "running rust benchmark for $g with $n qubits" && \
                for i in $(seq 1 $ITERS); do
                    ../../target/release/examples/circuits --qubits ${n} >> ${OUTPUT_DIR}/spinoza/${g}-${n}qubits
                done
            done
        done
}

bm_py_gates
bm_rust_gates

python plot.py --start_qubits $START_QUBITS --end_qubits $END_QUBITS --path_to_results $OUTPUT_DIR
