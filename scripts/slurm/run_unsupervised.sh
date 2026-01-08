#!/bin/bash
# =============================================================================
# Run only unsupervised learning (clustering) notebooks
# Assumes data processing is already complete
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting unsupervised learning (clustering) jobs..."

JOB_FE=$(sbatch --parsable "$SCRIPT_DIR/unsupervised/01_feature_engineering.sh")
echo "  Feature Engineering: $JOB_FE"

JOB_KMEANS=$(sbatch --parsable --dependency=afterok:$JOB_FE "$SCRIPT_DIR/unsupervised/02_kmeans.sh")
echo "  K-Means: $JOB_KMEANS"

JOB_DBSCAN=$(sbatch --parsable --dependency=afterok:$JOB_FE "$SCRIPT_DIR/unsupervised/03_dbscan.sh")
echo "  DBSCAN: $JOB_DBSCAN"

JOB_DBSCAN_CLUSTER=$(sbatch --parsable --dependency=afterok:$JOB_FE "$SCRIPT_DIR/unsupervised/04_dbscan_clustering.sh")
echo "  DBSCAN Clustering: $JOB_DBSCAN_CLUSTER"

JOB_CLARANS=$(sbatch --parsable --dependency=afterok:$JOB_FE "$SCRIPT_DIR/unsupervised/05_clarans.sh")
echo "  CLARANS: $JOB_CLARANS"

echo ""
echo "Unsupervised jobs submitted!"
echo "Note: Clustering jobs are computationally intensive and may take several hours."
