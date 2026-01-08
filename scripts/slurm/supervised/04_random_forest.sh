#!/bin/bash
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 07-00
#SBATCH -o /scratch/%u/DM/scripts/slurm/logs/random_forest_%j.out
#SBATCH -e /scratch/%u/DM/scripts/slurm/logs/random_forest_%j.err

# Random Forest with Bayesian optimization - heavy computation

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

# Run the random forest notebook
run_notebook "analysis/supervised/random_forest.ipynb"

exit $?
