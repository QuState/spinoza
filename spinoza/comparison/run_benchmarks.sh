#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -ne 4 ]]
then
  echo "Usage: $0 <start-qubits> <end-qubits> <iters> <qulacs-absolute-path>"
  exit 1
fi

START_QUBITS=$1
END_QUBITS=$2
ITERS=$3
QULACS_PATH=$4

OUTPUT_DIR=benchmark-data.$(date +"%Y.%m.%d.%H-%M-%S")
mkdir -p $OUTPUT_DIR/spinoza && mkdir $OUTPUT_DIR/qulacs

gates=("qcbm" "value_encoding" "rz" "rx" "ry" "x" "z" "p" "h")

bm_qulacs_gates () {
    export OMP_NUM_THREADS=$(nproc)
    make all QULACS_PATH=$QULACS_PATH -j $(nproc)

    for g in ${gates[@]}; do
        for n in $(seq $START_QUBITS $END_QUBITS); do
            echo "running qulacs benchmark for $g with $n qubits" && \
                for i in $(seq 1 $ITERS); do
                    ./qulacs_benchmark_${g} ${n} >> ${OUTPUT_DIR}/qulacs/${g}-${n}qubits
                done
            done
        done
}

bm_rust_gates () {
    cargo clean && cargo update && cargo build --release --examples

    for g in ${gates[@]}; do
        for n in $(seq $START_QUBITS $END_QUBITS); do
            echo "running rust benchmark for $g with $n qubits" && \
                for i in $(seq 1 $ITERS); do
                    ../../target/release/examples/${g} --qubits ${n} --threads $(nproc) >> ${OUTPUT_DIR}/spinoza/${g}-${n}qubits
                done
            done
        done
}

bm_rust_gates
bm_qulacs_gates

for g in ${gates[@]}; do
    python plot.py --start_qubits $START_QUBITS --end_qubits $END_QUBITS --path_to_results $OUTPUT_DIR --gate ${g}
done
