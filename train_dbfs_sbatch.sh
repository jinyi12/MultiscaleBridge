#!/bin/bash
#SBATCH -J dbfs_training  # Job name
#SBATCH -o dbfs_training.out  # Output file
#SBATCH -e dbfs_training.err  # Error file
#SBATCH -p gpu-common  # Partition (use gpu-common for GPU jobs)
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --mem=16G  # Request 16 GB of memory
#SBATCH --time=24:00:00  # Maximum runtime (24 hours)

# Load necessary modules (if needed)
module load Python/3.8.6

# Run your Python script
python /hpc/dctrl/jy384/MultiscaleBridge/dbfs/dbfs_grf_256_intOp_optimized.py