#!/bin/bash
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 03-00
#SBATCH -o /scratch/%u/DM/scripts/slurm/logs/kmeans_%j.out
#SBATCH -e /scratch/%u/DM/scripts/slurm/logs/kmeans_%j.err

# K-Means clustering with optimization

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

# Create logs directory if needed
mkdir -p "$LOGS_DIR"

# Initialize conda environment
init_conda

# Change to project directory
cd "$PROJECT_DIR"

# Run the K-Means notebook
run_notebook "analysis/unsupervised/kmeans.ipynb"

exit $?
