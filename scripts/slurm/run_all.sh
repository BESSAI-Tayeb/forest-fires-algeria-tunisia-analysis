#!/bin/bash
# =============================================================================
# Master script to submit all SLURM jobs with proper dependencies
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

echo "=============================================="
echo "Submitting SLURM jobs for Fire Risk Analysis"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Conda environment: $CONDA_ENV"
echo ""

# =============================================================================
# Phase 1: Data Processing (sequential - each depends on previous)
# =============================================================================
echo "Phase 1: Data Processing"
echo "------------------------"

# Fire processing (no dependencies)
JOB_FIRE=$(sbatch --parsable "$SCRIPT_DIR/data_processing/01_fire.sh")
echo "  Submitted fire processing: Job $JOB_FIRE"

# Landcover (depends on fire)
JOB_LANDCOVER=$(sbatch --parsable --dependency=afterok:$JOB_FIRE "$SCRIPT_DIR/data_processing/02_landcover.sh")
echo "  Submitted landcover processing: Job $JOB_LANDCOVER (after $JOB_FIRE)"

# Elevation, Soil, Climate can run in parallel after fire
JOB_ELEVATION=$(sbatch --parsable --dependency=afterok:$JOB_FIRE "$SCRIPT_DIR/data_processing/03_elevation.sh")
echo "  Submitted elevation processing: Job $JOB_ELEVATION (after $JOB_FIRE)"

JOB_SOIL=$(sbatch --parsable --dependency=afterok:$JOB_FIRE "$SCRIPT_DIR/data_processing/04_soil.sh")
echo "  Submitted soil processing: Job $JOB_SOIL (after $JOB_FIRE)"

JOB_CLIMATE=$(sbatch --parsable --dependency=afterok:$JOB_FIRE "$SCRIPT_DIR/data_processing/05_climate.sh")
echo "  Submitted climate processing: Job $JOB_CLIMATE (after $JOB_FIRE)"

# Merge (depends on all feature extractions)
JOB_MERGE=$(sbatch --parsable --dependency=afterok:$JOB_ELEVATION:$JOB_SOIL:$JOB_CLIMATE "$SCRIPT_DIR/data_processing/06_merge.sh")
echo "  Submitted merge processing: Job $JOB_MERGE (after elevation, soil, climate)"

# Sampling (depends on merge)
JOB_SAMPLING=$(sbatch --parsable --dependency=afterok:$JOB_MERGE "$SCRIPT_DIR/data_processing/07_sampling.sh")
echo "  Submitted sampling processing: Job $JOB_SAMPLING (after $JOB_MERGE)"

echo ""

# =============================================================================
# Phase 2: Supervised Learning (after data processing)
# =============================================================================
echo "Phase 2: Supervised Learning"
echo "----------------------------"

# Feature engineering (after sampling)
JOB_FE_SUP=$(sbatch --parsable --dependency=afterok:$JOB_SAMPLING "$SCRIPT_DIR/supervised/01_feature_engineering.sh")
echo "  Submitted supervised feature engineering: Job $JOB_FE_SUP"

# Model training (after feature engineering)
JOB_MODEL=$(sbatch --parsable --dependency=afterok:$JOB_FE_SUP "$SCRIPT_DIR/supervised/02_model_training.sh")
echo "  Submitted model training: Job $JOB_MODEL"

# Individual models can run in parallel after feature engineering
JOB_DT=$(sbatch --parsable --dependency=afterok:$JOB_FE_SUP "$SCRIPT_DIR/supervised/03_decision_tree.sh")
echo "  Submitted decision tree: Job $JOB_DT"

JOB_RF=$(sbatch --parsable --dependency=afterok:$JOB_FE_SUP "$SCRIPT_DIR/supervised/04_random_forest.sh")
echo "  Submitted random forest: Job $JOB_RF"

JOB_KNN=$(sbatch --parsable --dependency=afterok:$JOB_FE_SUP "$SCRIPT_DIR/supervised/05_knn.sh")
echo "  Submitted KNN: Job $JOB_KNN"

echo ""

# =============================================================================
# Phase 3: Unsupervised Learning (after data processing)
# =============================================================================
echo "Phase 3: Unsupervised Learning"
echo "------------------------------"

# Feature engineering (after sampling)
JOB_FE_UNSUP=$(sbatch --parsable --dependency=afterok:$JOB_SAMPLING "$SCRIPT_DIR/unsupervised/01_feature_engineering.sh")
echo "  Submitted unsupervised feature engineering: Job $JOB_FE_UNSUP"

# Clustering algorithms can run in parallel after feature engineering
JOB_KMEANS=$(sbatch --parsable --dependency=afterok:$JOB_FE_UNSUP "$SCRIPT_DIR/unsupervised/02_kmeans.sh")
echo "  Submitted K-Means: Job $JOB_KMEANS"

JOB_DBSCAN=$(sbatch --parsable --dependency=afterok:$JOB_FE_UNSUP "$SCRIPT_DIR/unsupervised/03_dbscan.sh")
echo "  Submitted DBSCAN: Job $JOB_DBSCAN"

JOB_DBSCAN_CLUSTER=$(sbatch --parsable --dependency=afterok:$JOB_FE_UNSUP "$SCRIPT_DIR/unsupervised/04_dbscan_clustering.sh")
echo "  Submitted DBSCAN clustering: Job $JOB_DBSCAN_CLUSTER"

JOB_CLARANS=$(sbatch --parsable --dependency=afterok:$JOB_FE_UNSUP "$SCRIPT_DIR/unsupervised/05_clarans.sh")
echo "  Submitted CLARANS: Job $JOB_CLARANS"

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Job summary:"
echo "  Data Processing:    $JOB_FIRE, $JOB_LANDCOVER, $JOB_ELEVATION, $JOB_SOIL, $JOB_CLIMATE, $JOB_MERGE, $JOB_SAMPLING"
echo "  Supervised:         $JOB_FE_SUP, $JOB_MODEL, $JOB_DT, $JOB_RF, $JOB_KNN"
echo "  Unsupervised:       $JOB_FE_UNSUP, $JOB_KMEANS, $JOB_DBSCAN, $JOB_DBSCAN_CLUSTER, $JOB_CLARANS"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in: $SCRIPT_DIR/logs/"
