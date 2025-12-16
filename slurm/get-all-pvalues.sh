#!/usr/bin/env sh

#SBATCH --ntasks 1
#SBATCH --array=1-100%50
#SBATCH --nodes=1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 8G
#SBATCH --partition dri.q
#SBATCH --account hazboun
#SBATCH --time 00:30:00
#SBATCH --job-name get-all-pvalues
#SBATCH --mail-user=wrightd2@oregonstate.edu
#SBATCH --mail-type=ALL
#SBATCH --output ./logs/get-all-pvalues-%A-%a-stdout.log
#SBATCH --error ./logs/get-all-pvalues-%A-%a-stderr.log

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
GAMMA=`echo "13 / 3" | $PIXI run bc -l`

$PIXI run python ../discovery_os_gx2.py get-pvalues-five-pta \
    --data-dir "$DATADIR" \
    --gw-log10-a -15.7 \
    --gw-gamma $GAMMA \
    --make-plots \
    --do-spna-max-likelihood \
    --rn-comp 30 \
    --gw-comp 7
