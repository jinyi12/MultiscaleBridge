#!/bin/bash
#SBATCH -J dbfs_training  # Job name
#SBATCH -o dbfs_training_%j.out  # Output file with job ID
#SBATCH -e dbfs_training_%j.err  # Error file with job ID
#SBATCH -p scavenger-gpu  # Partition (use gpu-common for GPU jobs)
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --mem=32G  # Request 32 GB of memory
#SBATCH --time=24:00:00  # Maximum runtime (24 hours)
#SBATCH --mail-type=BEGIN,END,FAIL     # Send email on job start, end, and fail
#SBATCH --mail-user=jy384@duke.edu  # Email address to send notifications


# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Print GPU information
nvidia-smi

# Load your conda environment
source /hpc/dctrl/jy384/miniconda3/etc/profile.d/conda.sh
conda activate myenv
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Move into the directory
cd /hpc/dctrl/jy384/MultiscaleBridge/dbfs

# Check directory structure
echo "Current working directory:"
pwd
echo "Checking for Data directory:"
ls -l ../Data
echo "Contents of Data directory:"
ls -l ../Data/*.npy

# Run Python script
echo "Starting training script..."
python BM2_FS_grf_256_intOp_optimized.py --intOp_scale_factor 10

# Print end time
echo "End time: $(date)" 