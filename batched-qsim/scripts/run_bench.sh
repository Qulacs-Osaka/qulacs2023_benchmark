#!/bin/bash

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo -e "qubits\tsamples\tbatch_size\truntime [s]"

for qubits in $(seq 3 15)
    do
    for samples in 10000 20000 50000
    do
        for batch_size in 1000 2000 5000 10000 20000 50000
        do
            if [ $samples -lt $batch_size ]; then
                continue
            fi
            echo -n -e "$qubits\t$samples\t$batch_size\t"
            build/qsim-bench --qubits $qubits --samples $samples --batch-size $batch_size
        done
    done
done
