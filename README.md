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

- **Merged Dataset**: `results/intermediate/fires_merged_all_features.csv`

  - Total Features: 42 (3 base + 12 climate + 4 elevation + 23 soil)
  - Total Samples: 5,978 rows
  - No Missing Values: All NaN values handled during preprocessing

- **Engineered Features**: `results/features/fires_engineered_features.csv`

  - Enhanced feature set with domain-specific transformations

- **Balanced Dataset**: `results/features/balanced_dataset.csv`

  - Balanced class distribution for model training

- **Imbalanced Dataset**: `results/features/imbalanced_dataset.csv`
  - Original class distribution preserved

## 🗂️ Project Structure
