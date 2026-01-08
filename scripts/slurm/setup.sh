#!/bin/bash
# =============================================================================
# Setup script for NYU HPC environment
# Run this once before submitting jobs
# =============================================================================

set -e

echo "=============================================="
echo "Setting up HPC environment for Fire Analysis"
echo "=============================================="

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Create required directories
echo "Creating directories..."
mkdir -p "$LOGS_DIR"
mkdir -p "$PROJECT_DIR/results/data"
mkdir -p "$PROJECT_DIR/results/intermediate"
mkdir -p "$PROJECT_DIR/results/models"

# Make all scripts executable
echo "Making scripts executable..."
chmod +x "$SCRIPT_DIR"/*.sh
chmod +x "$SCRIPT_DIR"/data_processing/*.sh
chmod +x "$SCRIPT_DIR"/supervised/*.sh
chmod +x "$SCRIPT_DIR"/unsupervised/*.sh

# Load conda module
echo "Loading conda module..."
module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"

# Check if environment exists
if conda env list | grep -q "^$CONDA_ENV "; then
    echo "Conda environment '$CONDA_ENV' found."
    conda activate "$CONDA_ENV"
else
    echo "Creating conda environment '$CONDA_ENV'..."
    conda create -n "$CONDA_ENV" python=3.10 -y
    conda activate "$CONDA_ENV"
    
    # Install required packages
    echo "Installing required packages..."
    pip install --upgrade pip
    
    # Core packages
    pip install pandas numpy scipy matplotlib seaborn
    
    # Geospatial packages
    pip install geopandas shapely fiona pyproj rasterio
    
    # Machine learning
    pip install scikit-learn scikit-optimize
    
    # Notebook execution
    pip install jupyter nbconvert papermill
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Your configuration:"
echo "  NetID: $NYU_NET_ID"
echo "  Project: $PROJECT_DIR"
echo "  Logs: $LOGS_DIR"
echo "  Conda env: $CONDA_ENV"
echo ""
echo "Next steps:"
echo "  1. Copy your project files to: $PROJECT_DIR"
echo "  2. Run: ./run_all.sh"
echo "  3. Monitor: ./utils.sh status"
