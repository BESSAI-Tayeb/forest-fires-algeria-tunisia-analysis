#!/bin/bash
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02-00
#SBATCH -o /scratch/%u/DM/scripts/slurm/logs/elevation_%j.out
#SBATCH -e /scratch/%u/DM/scripts/slurm/logs/elevation_%j.err

# Elevation processing notebook

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

# Create logs directory if needed
mkdir -p "$LOGS_DIR"

# Initialize conda environment
init_conda

# Change to project directory
cd "$PROJECT_DIR"

# Run the elevation notebook
run_notebook "analysis/analysis/elevation.ipynb"

exit $?
