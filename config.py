"""
Project Configuration File

This module centralizes all path configurations for the wildfire prediction project.
Import this file in notebooks to access consistent paths across the project.

Usage:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Adjust based on notebook location
    from config import *
"""

from pathlib import Path

# =============================================================================
# BASE PATHS
# =============================================================================
# Project root directory (where this config.py file is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# =============================================================================
# DATASET PATHS
# =============================================================================
DATASET_DIR = PROJECT_ROOT / 'dataset'

# Fire dataset
FIRE_DATASET_DIR = DATASET_DIR / 'fire_dataset'
ALGERIA_FIRES_CSV = FIRE_DATASET_DIR / 'algeria.csv'
TUNISIA_FIRES_CSV = FIRE_DATASET_DIR / 'tunisia.csv'

# Climate dataset
CLIMATE_DATASET_DIR = DATASET_DIR / 'climate_dataset'
PREC_DIR = CLIMATE_DATASET_DIR / 'prec'
TMAX_DIR = CLIMATE_DATASET_DIR / 'tmax'
TMIN_DIR = CLIMATE_DATASET_DIR / 'tmin'

# Elevation dataset
ELEVATION_DATASET_DIR = DATASET_DIR / 'elevation_dataset'
ELEVATION_RASTER = ELEVATION_DATASET_DIR / 'be15_grd'
ELEVATION_GRD_DIR = ELEVATION_DATASET_DIR / 'elevation_grd'

# Land cover dataset
LAND_COVER_DIR = DATASET_DIR / 'land_cover_dataset'
ALGERIA_LC_DIR = LAND_COVER_DIR / 'algeria'
TUNISIA_LC_DIR = LAND_COVER_DIR / 'tunisia'
ALGERIA_SHP = ALGERIA_LC_DIR / 'dza_gc_adg.shp'
TUNISIA_SHP = TUNISIA_LC_DIR / 'tun_gc_adg.shp'

# Soil dataset
SOIL_DATASET_DIR = DATASET_DIR / 'soil_dataset'
HWSD_MDB = SOIL_DATASET_DIR / 'HWSD2.mdb'
HWSD_BIL = SOIL_DATASET_DIR / 'HWSD2.bil'

# =============================================================================
# RESULTS PATHS (Organized)
# =============================================================================
RESULTS_DIR = PROJECT_ROOT / 'results'

# Base data files
RESULTS_DATA_DIR = RESULTS_DIR / 'data'
FIRES_CSV = RESULTS_DATA_DIR / 'fires.csv'

FIRES_SUPERVISED_CSV = RESULTS_DATA_DIR / 'fires_supervised.csv'
FIRES_UNSUPERVISED_CSV = RESULTS_DATA_DIR / 'fires_unsupervised.csv'
FIRES_WITH_ARTIFICIAL_SUPERVISED = RESULTS_DATA_DIR / 'fires_with_artificial_supervised.csv'
FIRES_WITH_ARTIFICIAL_UNSUPERVISED = RESULTS_DATA_DIR / 'fires_with_artificial_unsupervised.csv'

SEA_CSV = RESULTS_DATA_DIR / 'sea.csv'
ALGERIA_BOUNDARY = RESULTS_DATA_DIR / 'algeria_boundary.geojson'
TUNISIA_BOUNDARY = RESULTS_DATA_DIR / 'tunisia_boundary.geojson'
ALGERIA_LC_CLEAN = RESULTS_DATA_DIR / 'algeria_landcover_clean.geojson'
TUNISIA_LC_CLEAN = RESULTS_DATA_DIR / 'tunisia_landcover_clean.geojson'
MERGED_LC_CLEAN = RESULTS_DATA_DIR / 'merged_landcover_clean.geojson'

# Intermediate processing files
RESULTS_INTERMEDIATE_DIR = RESULTS_DIR / 'intermediate'
FIRES_WITH_CLIMATE = RESULTS_INTERMEDIATE_DIR / 'fires_with_climate.csv'
FIRES_WITH_ELEVATION = RESULTS_INTERMEDIATE_DIR / 'fires_with_elevation_features.csv'
FIRES_WITH_SOIL = RESULTS_INTERMEDIATE_DIR / 'fires_with_soil_features.csv'
FIRES_WITH_SOIL_IDS = RESULTS_INTERMEDIATE_DIR / 'fires_with_soil_ids.csv'
FIRES_MERGED = RESULTS_INTERMEDIATE_DIR / 'fires_merged_all_features.csv'
FIRES_MERGED_BALANCED = RESULTS_INTERMEDIATE_DIR / 'fires_merged_all_features_balanced.csv'

# Feature datasets (final)
RESULTS_FEATURES_DIR = RESULTS_DIR / 'features'
BALANCED_DATASET = RESULTS_FEATURES_DIR / 'balanced_dataset.csv'
IMBALANCED_DATASET = RESULTS_FEATURES_DIR / 'imbalanced_dataset.csv'
FIRES_ENGINEERED = RESULTS_FEATURES_DIR / 'fires_engineered_features.csv'
FIRES_ENGINEERED_PCA = RESULTS_FEATURES_DIR / 'fires_engineered_pca.csv'  # PCA-transformed (optional)
SELECTED_FEATURES_TXT = RESULTS_FEATURES_DIR / 'selected_features.txt'
FEATURE_IMPORTANCE_CSV = RESULTS_FEATURES_DIR / 'feature_importance_scores.csv'

# Model outputs
RESULTS_MODELS_DIR = RESULTS_DIR / 'models'
MODEL_PERFORMANCE = RESULTS_MODELS_DIR / 'model_performance.csv'
CONFUSION_MATRIX = RESULTS_MODELS_DIR / 'confusion_matrix.csv'
MODEL_FEATURE_IMPORTANCE = RESULTS_MODELS_DIR / 'feature_importance.csv'
TRAIN_DATASET = RESULTS_MODELS_DIR / 'train_dataset.csv'
TEST_DATASET = RESULTS_MODELS_DIR / 'test_dataset.csv'
TEST_PREDICTIONS = RESULTS_MODELS_DIR / 'test_predictions.csv'

# Processing logs
RESULTS_LOGS_DIR = RESULTS_DIR / 'logs'
ELEVATION_LOG = RESULTS_LOGS_DIR / 'elevation_processing_summary.txt'
SOIL_LOG = RESULTS_LOGS_DIR / 'soil_processing_summary.txt'

# Clipped rasters
CLIPPED_RASTERS_DIR = RESULTS_DIR / 'clipped_rasters'

# =============================================================================
# MODELS DIRECTORY
# =============================================================================
MODELS_DIR = PROJECT_ROOT / 'models'

# =============================================================================
# ANALYSIS DIRECTORY (Organized)
# =============================================================================
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'

# Subdirectories
ANALYSIS_DATA_PROCESSING_DIR = ANALYSIS_DIR / 'analysis'
ANALYSIS_SUPERVISED_DIR = ANALYSIS_DIR / 'supervised'
ANALYSIS_UNSUPERVISED_DIR = ANALYSIS_DIR / 'unsupervised'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def ensure_dirs():
    """Create all output directories if they don't exist."""
    dirs = [
        RESULTS_DATA_DIR,
        RESULTS_INTERMEDIATE_DIR,
        RESULTS_FEATURES_DIR,
        RESULTS_MODELS_DIR,
        RESULTS_LOGS_DIR,
        CLIPPED_RASTERS_DIR,
        ANALYSIS_DATA_PROCESSING_DIR,
        ANALYSIS_SUPERVISED_DIR,
        ANALYSIS_UNSUPERVISED_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def get_project_root():
    """Return the project root directory."""
    return PROJECT_ROOT

# =============================================================================
# MULTITHREADING CONFIGURATION
# =============================================================================
import os
import multiprocessing

def configure_threading(max_threads=None):
    """Set thread env variables so numeric libs leverage multiple CPU cores."""
    if max_threads is None:
        max_threads = multiprocessing.cpu_count()
    thread_env_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for var in thread_env_vars:
        os.environ[var] = str(max_threads)
    print(f"Threading configured to use up to {max_threads} CPU cores.")
    return max_threads

# =============================================================================
# AUTO-INITIALIZATION
# =============================================================================
# Ensure directories exist when config is imported
ensure_dirs()
