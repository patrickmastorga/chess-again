#!/bin/bash
#SBATCH -J v1-profile
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
# --gres asks for 8 GPUs of any type; -C HX00 narrows to H100 or H200, both
# 8-GPU SXM5 nodes. HX00 roughly doubles the matching node pool over H100
# alone, and 80GB vs 142GB does not matter at this model size.
#SBATCH --gres=gpu:8
#SBATCH -C HX00
# PACE's recommendation for HGX nodes: the per-GPU share of the node's 2048GB,
# so 8 GPUs claims the whole machine's memory. The default would be 1GB/core.
#SBATCH --mem-per-gpu=224GB
#SBATCH -t1:00:00
#SBATCH -oprofile-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# -e: stop at the first failure. -u: an unset variable is an error
set -eu

cd "$SLURM_SUBMIT_DIR"

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
# Record the devices Slurm actually granted
nvidia-smi -L
echo

# Fail early and clearly if the build is missing
BIN=target/release/v1_profile_ddp
if [[ ! -x $BIN ]]; then
    echo "missing $BIN: build it first with" >&2
    echo "  cargo build --release --features cuda -p chess-again-train --bin v1_profile_ddp" >&2
    exit 1
fi

"$BIN"

echo
echo "=== done ==="
