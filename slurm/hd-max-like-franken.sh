#!/usr/bin/env sh

#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --array=1-100%10
#SBATCH --nodes=1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 8G
#SBATCH --partition dri.q
#SBATCH --account hazboun
#SBATCH --time 00:30:00
#SBATCH --job-name hd-max-like-franken
#SBATCH --mail-user=wrightd2@oregonstate.edu
#SBATCH --mail-type=ALL
#SBATCH --output ./logs/hd-max-like-franken-%A-%a-stdout.log
#SBATCH --error ./logs/hd-max-like-franken-%A-%a-stderr.log

mkdir -p ../logs/

# Detect the CPU architecture
ARCH=$(uname -m)

# Choose the appropriate binary based on architecture
if [[ "$ARCH" == "arm"* || "$ARCH" == "aarch64" ]]; then
    PIXI="pixi-arm"
elif [[ "$ARCH" == "x86_64" || "$ARCH" == "amd64" ]]; then
    PIXI="pixi"
else
    echo "Unsupported architecture: $ARCH" >&2
    exit 1
fi

DATADIR="$HOME/novus/codes/frankenstat-simulations/outputs/2025-10-18/realization-$SLURM_ARRAY_TASK_ID"

$PIXI run python ../analysis.py run-array-max-like \
    --data-dir "$DATADIR/franken_psrs" \
    --svi-samples 50000 \
    --batch-size 1000 \
    --psrs-prefix franken \
    --rn-comp 30 \
    --gw-comp 7
