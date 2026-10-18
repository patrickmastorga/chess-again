#!/bin/bash
#SBATCH -J v1-profile
#SBATCH -N1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:H100:8
#SBATCH -C H100
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
