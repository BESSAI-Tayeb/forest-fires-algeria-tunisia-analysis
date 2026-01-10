# Fire Risk Prediction in Algeria and Tunisia

A comprehensive data mining project for predicting wildfire occurrences using environmental features including climate, elevation, soil properties, and land cover data.

## 📋 Project Overview

This project analyzes fire risk patterns in Algeria and Tunisia by integrating multiple environmental datasets and building machine learning models for fire prediction. The analysis combines geospatial data processing, exploratory data analysis, feature engineering, and custom machine learning implementations (both supervised and unsupervised).

### Key Objectives

- Process and integrate multi-source environmental data (climate, elevation, soil, land cover)
- Perform comprehensive exploratory data analysis (EDA)
- Engineer meaningful features for fire prediction
- Build custom machine learning models from scratch (minimal sklearn dependencies)
- Apply supervised learning for fire occurrence prediction
- Apply unsupervised learning for pattern discovery and clustering

## 📊 Dataset Description

### Fire Dataset

- **Source**: FIRMS (Fire Information for Resource Management System)
- **Coverage**: Algeria and Tunisia
- **Records**: 5,978 fire points (after sea removal)
- **Classes**: Binary (0 = Non-Fire, 1 = Fire)

### Environmental Features

#### 1. Climate Data (12 features)

- **Source**: CHIRPS (Precipitation) + TerraClimate (Temperature)
- **Temporal Resolution**: Monthly data aggregated by season
- **Features**:
  - Seasonal precipitation (winter, spring, summer, autumn)
  - Seasonal max temperature (4 seasons)
  - Seasonal min temperature (4 seasons)

#### 2. Elevation Data (4 features)

- **Source**: GMTED2010 (Global Multi-resolution Terrain Elevation Data)
- **Features**:
  - Elevation (meters)
  - Slope (degrees)
  - Aspect (direction)
  - Roughness (terrain variation)

#### 3. Soil Properties (23 features)

- **Source**: HWSD v2.0 (Harmonized World Soil Database)
- **Features**:
  - Physical properties: SAND, SILT, CLAY, COARSE, BULK density
  - Chemical properties: ORG_CARBON, PH_WATER, TOTAL_N, CN_RATIO
  - Fertility indicators: CEC_SOIL, CEC_CLAY, TEB, BSAT
  - Other: TEXTURE_USDA, TEXTURE_SOTER, GYPSUM, ELEC_COND, etc.

#### 4. Land Cover Data

- **Source**: FAO Global Land Cover (Algeria & Tunisia)
- **Type**: Categorical land use/land cover classifications

### Processed Datasets

The project generates separate datasets for supervised and unsupervised learning:

#### Supervised Learning Datasets
- **Location**: `results/intermediate/supervised/`
- **Base Dataset**: `fires_with_artificial_supervised.csv` (lat > 33° for balanced classes)
- **Merged Dataset**: `fires_merged_all_features.csv` - All features combined

#### Unsupervised Learning Datasets
- **Location**: `results/intermediate/unsupervised/`
- **Base Dataset**: `fires_with_artificial_unsupervised.csv` (full geographic coverage)
- **Merged Dataset**: `fires_merged_all_features.csv` - All features combined

#### Feature Composition
- Total Features: 42 (3 base + 12 climate + 4 elevation + 23 soil)
- No Missing Values: All NaN values handled during preprocessing

#### Additional Output Files
- **Engineered Features**: `results/features/fires_engineered_features.csv`
- **Balanced Dataset**: `results/features/balanced_dataset.csv`
- **Imbalanced Dataset**: `results/features/imbalanced_dataset.csv`

## 🗂️ Project Structure

```
DM/
├── config.py                    # Centralized path configuration
├── README.md
├── requirements.txt
├── analysis/
│   ├── analysis/                # Data processing notebooks
│   │   ├── landcover.ipynb      # 1️⃣ Land cover data processing
│   │   ├── fire.ipynb           # 2️⃣ Fire dataset analysis & artificial points
│   │   ├── climate.ipynb        # 3️⃣ Climate feature extraction
│   │   ├── elevation.ipynb      # 4️⃣ Elevation feature extraction
│   │   ├── soil.ipynb           # 5️⃣ Soil feature extraction
│   │   ├── merge.ipynb          # 6️⃣ Merge all datasets
│   │   └── sampling.ipynb       # Additional sampling utilities
│   ├── supervised/              # Supervised learning notebooks
│   │   ├── feature_engineering_supervised.ipynb
│   │   ├── model_training.ipynb
│   │   ├── decision_tree.ipynb
│   │   ├── random_forest.ipynb
│   │   └── knn.ipynb
│   └── unsupervised/            # Unsupervised learning notebooks
│       ├── feature_engineering_unsupervised.ipynb
│       ├── kmeans.ipynb
│       ├── dbscan.ipynb
│       └── clarans.ipynb
├── dataset/                     # Raw input data
│   ├── fire_dataset/
│   ├── climate_dataset/
│   ├── elevation_dataset/
│   ├── land_cover_dataset/
│   └── soil_dataset/
├── results/                     # Processed outputs
│   ├── data/                    # Base processed data
│   │   ├── fires.csv
│   │   ├── fires_supervised.csv
│   │   ├── fires_unsupervised.csv
│   │   ├── fires_with_artificial_supervised.csv
│   │   ├── fires_with_artificial_unsupervised.csv
│   │   └── *.geojson            # Land cover boundaries
│   ├── intermediate/            # Feature extraction outputs
│   │   ├── supervised/          # Supervised learning pipeline
│   │   │   ├── fires_with_climate.csv
│   │   │   ├── fires_with_elevation_features.csv
│   │   │   ├── fires_with_soil_features.csv
│   │   │   └── fires_merged_all_features.csv
│   │   └── unsupervised/        # Unsupervised learning pipeline
│   │       ├── fires_with_climate.csv
│   │       ├── fires_with_elevation_features.csv
│   │       ├── fires_with_soil_features.csv
│   │       └── fires_merged_all_features.csv
│   ├── features/                # Engineered features
│   └── models/                  # Trained model outputs
├── models/                      # Custom ML implementations
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── knn.py
│   ├── kmeans.py
│   └── dbscan.py
├── report/                      # LaTeX report files
└── scripts/                     # Automation scripts
    └── slurm/                   # HPC cluster scripts
```

## 🚀 Notebook Execution Order

Execute the data processing notebooks in the following order:

| Step | Notebook | Description | Output |
|------|----------|-------------|--------|
| 1️⃣ | `analysis/analysis/landcover.ipynb` | Process land cover shapefiles, create boundaries | `results/data/*.geojson` |
| 2️⃣ | `analysis/analysis/fire.ipynb` | Load fire data, generate artificial non-fire points, split supervised/unsupervised | `results/data/fires_*.csv` |
| 3️⃣ | `analysis/analysis/climate.ipynb` | Extract climate features (precipitation, temperature) | `results/intermediate/*/fires_with_climate.csv` |
| 4️⃣ | `analysis/analysis/elevation.ipynb` | Extract elevation features (slope, aspect, roughness) | `results/intermediate/*/fires_with_elevation_features.csv` |
| 5️⃣ | `analysis/analysis/soil.ipynb` | Extract soil properties from HWSD database | `results/intermediate/*/fires_with_soil_features.csv` |
| 6️⃣ | `analysis/analysis/merge.ipynb` | Merge all features into final datasets | `results/intermediate/*/fires_merged_all_features.csv` |

### Execution Notes

- **Dependencies**: Each notebook depends on outputs from previous notebooks
- **Supervised vs Unsupervised**: Steps 3-6 process both supervised and unsupervised datasets in parallel
- **Configuration**: All paths are managed via `config.py` - import it at the start of each notebook
- **Sea Point Removal**: Fire notebook removes sea points (identified in soil processing) from all datasets

### After Data Processing

Once data processing is complete, proceed with:

**For Supervised Learning:**
1. `analysis/supervised/feature_engineering_supervised.ipynb`
2. `analysis/supervised/model_training.ipynb` (or individual model notebooks)

**For Unsupervised Learning:**
1. `analysis/unsupervised/feature_engineering_unsupervised.ipynb`
2. `analysis/unsupervised/kmeans.ipynb`, `dbscan.ipynb`, or `clarans.ipynb`
