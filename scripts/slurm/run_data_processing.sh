#!/bin/bash
# =============================================================================
# Run only data processing notebooks
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting data processing jobs..."

JOB_FIRE=$(sbatch --parsable "$SCRIPT_DIR/data_processing/01_fire.sh")
echo "  Fire: $JOB_FIRE"

JOB_LANDCOVER=$(sbatch --parsable --dependency=afterok:$JOB_FIRE "$SCRIPT_DIR/data_processing/02_landcover.sh")
echo "  Landcover: $JOB_LANDCOVER"

JOB_ELEVATION=$(sbatch --parsable --dependency=afterok:$JOB_FIRE "$SCRIPT_DIR/data_processing/03_elevation.sh")
JOB_SOIL=$(sbatch --parsable --dependency=afterok:$JOB_FIRE "$SCRIPT_DIR/data_processing/04_soil.sh")
JOB_CLIMATE=$(sbatch --parsable --dependency=afterok:$JOB_FIRE "$SCRIPT_DIR/data_processing/05_climate.sh")
echo "  Elevation: $JOB_ELEVATION, Soil: $JOB_SOIL, Climate: $JOB_CLIMATE"

JOB_MERGE=$(sbatch --parsable --dependency=afterok:$JOB_ELEVATION:$JOB_SOIL:$JOB_CLIMATE "$SCRIPT_DIR/data_processing/06_merge.sh")
echo "  Merge: $JOB_MERGE"

JOB_SAMPLING=$(sbatch --parsable --dependency=afterok:$JOB_MERGE "$SCRIPT_DIR/data_processing/07_sampling.sh")
echo "  Sampling: $JOB_SAMPLING"

echo ""
echo "Data processing jobs submitted. Final job: $JOB_SAMPLING"
