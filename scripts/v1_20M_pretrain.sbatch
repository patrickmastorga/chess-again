#!/bin/bash
#SBATCH -J v1-20M-pretrain
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Request any device count from 1 to 4 GPUs. Slurm picks the count with the
# best chance of starting soon, and the training binary uses whatever it gets.
#SBATCH --gres=gpu:1-4

# -C HX00 matches H100 or H200; the alternation widens the pool to the other
# Nvidia types that are reasonable here. Everything listed has well over the
# memory a 20M-parameter model in fp32 needs.
#SBATCH -C 'HX00|L40S|A100-80GB|A100-40GB'

# Not 224GB/GPU: that is PACE's H100-node figure, and the 768GB L40S nodes
# cannot host a 1-4 GPU request at that size. This model needs a few GB per
# GPU, so 32GB keeps every class in the constraint schedulable.
#SBATCH --mem-per-gpu=32GB
#SBATCH -t16:00:00
#SBATCH -ov1-20M-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pastorga3@gatech.edu

# -e: stop at the first failure. -u: an unset variable is an error
set -eu

cd "$SLURM_SUBMIT_DIR"

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
date
echo "granted GPUs: ${SLURM_GPUS_ON_NODE:-unknown}"
# Record the devices Slurm actually granted
nvidia-smi -L
echo

# Fail early and clearly if the build is missing
BIN=target/release/v1_20M_pretrain
if [[ ! -x $BIN ]]; then
    echo "missing $BIN: build it first with" >&2
    echo "  cargo build --release --features cuda -p chess-again-train --bin v1_20M_pretrain" >&2
    exit 1
fi

# The binary enumerates the allocated CUDA devices itself and trains on the
# most of them that divide its batch size evenly.
"$BIN"

echo
echo "=== done ==="
date
