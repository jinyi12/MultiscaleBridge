#!/bin/bash
#SBATCH -J BM2_eval            # Job name
#SBATCH -o BM2_eval_%j.out      # Output file (includes Job ID)
#SBATCH -e BM2_eval_%j.err      # Error file (includes Job ID)
#SBATCH -p scavenger-gpu           # Partition (use gpu-common for GPU jobs)
#SBATCH --gres=gpu:1    # Request 1 GPU
#SBATCH --mem=32G               # Request 32GB of memory
#SBATCH --time=08:00:00         # Maximum runtime (8 hours)
#SBATCH --mail-type=END,FAIL    # Email on job end and failure
#SBATCH --mail-user=jy384@duke.edu  # Your email address

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
nvidia-smi

# Activate the conda environment
source /hpc/dctrl/jy384/miniconda3/etc/profile.d/conda.sh
conda activate myenv
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# If needed, change to your project root directory (uncomment the next line if required)
# cd /hpc/dctrl/jy384/MultiscaleBridge

# Set parameters (override these using environment variables if desired)
WANDB_NAME=${WANDB_NAME:-"sandy-sea-164"}
ARTIFACT_VERSION=${ARTIFACT_VERSION:-"v9"}
N_FIELDS=${N_FIELDS:-100}
N_SAMPLES=${N_SAMPLES:-500}
BATCH_SIZE=${BATCH_SIZE:-16}
DISCRETIZATION_STEPS=${DISCRETIZATION_STEPS:-30}
ENERGY=${ENERGY:-1.0}

echo "Evaluating run: $WANDB_NAME, version: $ARTIFACT_VERSION"
echo "Parameters: N_FIELDS=$N_FIELDS, N_SAMPLES=$N_SAMPLES, BATCH_SIZE=$BATCH_SIZE, DISCRETIZATION_STEPS=$DISCRETIZATION_STEPS, ENERGY=$ENERGY"

# Launch the evaluation script
python BM2_evaluate_forward_generation.py \
    --wandb_name $WANDB_NAME \
    --artifact_version $ARTIFACT_VERSION \
    --n_fields $N_FIELDS \
    --n_samples $N_SAMPLES \
    --batch_size $BATCH_SIZE \
    --discretization_steps $DISCRETIZATION_STEPS \
    --energy $ENERGY \
    --force_eval \
    > ${WANDB_NAME}_eval.log

echo "End time: $(date)" 