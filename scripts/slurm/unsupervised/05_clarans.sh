#!/bin/bash
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 07-00
#SBATCH -o /scratch/%u/DM/scripts/slurm/logs/clarans_%j.out
#SBATCH -e /scratch/%u/DM/scripts/slurm/logs/clarans_%j.err

# CLARANS clustering - very computationally intensive

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

# Use all available threads for heavy job
export OMP_NUM_THREADS=28

# Create logs directory if needed
mkdir -p "$LOGS_DIR"

# Initialize conda environment
init_conda

# Change to project directory
cd "$PROJECT_DIR"

# Run the CLARANS notebook
run_notebook "analysis/unsupervised/clarans.ipynb"

exit $?
