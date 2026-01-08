#!/bin/bash
# =============================================================================
# SLURM Configuration - Edit this file for your HPC cluster
# =============================================================================

# User settings
export NYU_NET_ID="${NYU_NET_ID:-$(whoami)}"  # Your NetID (auto-detect if not set)

# Cluster-specific settings
export PARTITION="compute"          # Partition name
export RESERVATION=""               # Reservation (e.g., "c2", leave empty if none)
export ACCOUNT=""                   # Your allocation account (leave empty if not required)
export QOS=""                       # Quality of Service (leave empty if not required)

# Project paths (using scratch directory)
export SCRATCH_DIR="/scratch/$NYU_NET_ID"
export PROJECT_DIR="$SCRATCH_DIR/DM"        # Path to the project directory on HPC
export LOGS_DIR="$PROJECT_DIR/scripts/slurm/logs"

# Conda environment
export CONDA_ENV="myenv"            # Name of your conda environment
export CONDA_MODULE="miniconda-nobashrc"  # Module to load for conda

# Parallelization settings
export OMP_NUM_THREADS="12"         # OpenMP threads (adjust based on job cpus)

# Default resource limits (can be overridden per job)
export DEFAULT_TIME="02-00"         # 2 days default (format: days-hours)
export DEFAULT_MEM="16G"            # 16GB RAM default
export DEFAULT_CPUS="8"             # 8 CPUs default

# Heavy job resource limits (for clustering algorithms)
export HEAVY_TIME="07-00"           # 7 days for heavy jobs
export HEAVY_MEM="64G"              # 64GB RAM for heavy jobs
export HEAVY_CPUS="28"              # 28 CPUs for heavy jobs (full node)

# Email notifications (optional)
export MAIL_USER=""                 # Your email address
export MAIL_TYPE="FAIL,END"         # When to send emails (NONE, BEGIN, END, FAIL, ALL)

# =============================================================================
# Helper function to initialize conda
# =============================================================================
init_conda() {
    # Load conda module
    module load "$CONDA_MODULE"
    
    # Initialize conda for bash
    eval "$(conda shell.bash hook)"
    
    # Activate environment
    conda activate "$CONDA_ENV"
    
    # Set parallelization
    export OMP_NUM_THREADS="$OMP_NUM_THREADS"
    
    # Print environment info
    echo "========================================"
    echo "Environment Info:"
    echo "  User: $NYU_NET_ID"
    echo "  Python: $(which python)"
    echo "  Conda env: $CONDA_ENV"
    echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
    echo "  Working dir: $(pwd)"
    echo "========================================"
}

# =============================================================================
# Helper function to run a notebook
# =============================================================================
run_notebook() {
    local notebook_path="$1"
    local output_path="${notebook_path%.ipynb}_executed.ipynb"
    
    echo "========================================"
    echo "Running notebook: $notebook_path"
    echo "Started at: $(date)"
    echo "========================================"
    
    # Run notebook using papermill or nbconvert
    if command -v papermill &> /dev/null; then
        papermill "$notebook_path" "$output_path" --no-progress-bar
    else
        jupyter nbconvert --to notebook --execute --inplace "$notebook_path" \
            --ExecutePreprocessor.timeout=None \
            --ExecutePreprocessor.allow_errors=False
    fi
    
    local exit_code=$?
    
    echo "========================================"
    echo "Finished at: $(date)"
    echo "Exit code: $exit_code"
    echo "========================================"
    
    return $exit_code
}

# Export functions for use in job scripts
export -f init_conda
export -f run_notebook
