#!/usr/bin/env sh

#SBATCH --ntasks 126
#SBATCH --array=1-100%2
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 2G
#SBATCH --partition dri.q
#SBATCH --account hazboun
#SBATCH --time 00:30:00
##SBATCH --ntasks 1
##SBATCH --array=1-100%2
##SBATCH --nodelist=cos-gh01
##SBATCH --mem 0
##SBATCH --exclusive
##SBATCH --gres=gpu:gh200:1
##SBATCH --partition cos-arm.q
##SBATCH --account ph
##SBATCH --time 00:30:00
#SBATCH --job-name spna-franken
#SBATCH --mail-user=wrightd2@oregonstate.edu
#SBATCH --mail-type=ALL
#SBATCH --output ./logs/spna-franken-%A-%a-stdout.log
#SBATCH --error ./logs/spna-franken-%A-%a-stderr.log

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

$PIXI run mpirun --mca pml ob1 -n $SLURM_NTASKS  python ../simulation.py spna-max-likelihood-parallel \
    --feather-dir "$DATADIR/franken_psrs" \
    --svi-samples 100000 \
    --svi-batch-size 1000 \
    --use-mpi \
    --psr-prefix "franken"


