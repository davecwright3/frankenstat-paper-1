#!/usr/bin/env sh

#SBATCH --ntasks 126
#SBATCH --array=1-100%2
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 2G
#SBATCH --partition dri.q
#SBATCH --account hazboun
#SBATCH --time 00:30:00
#SBATCH --job-name simulate-combined
#SBATCH --mail-user=wrightd2@oregonstate.edu
#SBATCH --mail-type=ALL
#SBATCH --output ./logs/simulate-combined-%A-%a-stdout.log
#SBATCH --error ./logs/simulate-combined-%A-%a-stderr.log


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

mkdir -p ../logs/

module purge
module load git
module load gnu12
module load openmpi4

DATADIR="$HOME/novus/codes/frankenstat-simulations/outputs/2025-10-18/realization-$SLURM_ARRAY_TASK_ID"

GAMMA=`echo "13 / 3" | pixi run bc -l`

$PIXI run mpirun --mca pml ob1 -n $SLURM_NTASKS  python ../simulation.py simulate-pull-from-priors-parallel \
    --data-dir "$DATADIR" \
    --npsr 126 \
    --gw-log10-a -15.7 \
    --gw-gamma $GAMMA \
    --no-ecorr-include \
    --wn-const \
    --rn-include \
    --rn-comp 30 \
    --rng-seed $SLURM_ARRAY_TASK_ID \
    --use-mpi
