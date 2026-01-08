#!/bin/bash
# =============================================================================
# Run only supervised learning notebooks
# Assumes data processing is already complete
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting supervised learning jobs..."

JOB_FE=$(sbatch --parsable "$SCRIPT_DIR/supervised/01_feature_engineering.sh")
echo "  Feature Engineering: $JOB_FE"

JOB_MODEL=$(sbatch --parsable --dependency=afterok:$JOB_FE "$SCRIPT_DIR/supervised/02_model_training.sh")
echo "  Model Training: $JOB_MODEL"

JOB_DT=$(sbatch --parsable --dependency=afterok:$JOB_FE "$SCRIPT_DIR/supervised/03_decision_tree.sh")
JOB_RF=$(sbatch --parsable --dependency=afterok:$JOB_FE "$SCRIPT_DIR/supervised/04_random_forest.sh")
JOB_KNN=$(sbatch --parsable --dependency=afterok:$JOB_FE "$SCRIPT_DIR/supervised/05_knn.sh")
echo "  Decision Tree: $JOB_DT, Random Forest: $JOB_RF, KNN: $JOB_KNN"

echo ""
echo "Supervised jobs submitted!"
