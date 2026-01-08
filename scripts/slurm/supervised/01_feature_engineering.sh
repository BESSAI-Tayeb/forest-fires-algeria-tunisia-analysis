#!/bin/bash
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01-00
#SBATCH -o /scratch/%u/DM/scripts/slurm/logs/fe_supervised_%j.out
#SBATCH -e /scratch/%u/DM/scripts/slurm/logs/fe_supervised_%j.err

# Feature engineering for supervised learning

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

# Create logs directory if needed
mkdir -p "$LOGS_DIR"

# Initialize conda environment
init_conda

# Change to project directory
cd "$PROJECT_DIR"

# Run the feature engineering notebook
run_notebook "analysis/supervised/feature_engineering_supervised.ipynb"

exit $?
