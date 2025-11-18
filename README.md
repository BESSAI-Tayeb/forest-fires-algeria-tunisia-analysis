# Fire Risk Prediction in Algeria and Tunisia

A comprehensive data mining project for predicting wildfire occurrences using environmental features including climate, elevation, soil properties, and land cover data.

## 📋 Project Overview

This project analyzes fire risk patterns in Algeria and Tunisia by integrating multiple environmental datasets and building machine learning models for fire prediction. The analysis combines geospatial data processing, exploratory data analysis, and custom machine learning implementations.

### Key Objectives
- Process and integrate multi-source environmental data (climate, elevation, soil, land cover)
- Perform comprehensive exploratory data analysis (EDA)
- Build custom machine learning models from scratch (no sklearn dependencies)
- Predict fire occurrence based on environmental features

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

### Merged Dataset
- **File**: `results/fires_merged_all_features.csv`
- **Total Features**: 42 (3 base + 12 climate + 4 elevation + 23 soil)
- **Total Samples**: 5,978 rows
- **No Missing Values**: All NaN values handled during preprocessing

## 🗂️ Project Structure

```
DM/
├── analysis/                    # Jupyter notebooks for data analysis
│   ├── climate.ipynb           # Climate data processing & EDA
│   ├── elevation.ipynb         # Elevation feature extraction
│   ├── fire.ipynb              # Fire dataset processing
│   ├── landcover.ipynb         # Land cover analysis
│   ├── merge.ipynb             # Dataset integration
│   └── soil.ipynb              # Soil properties extraction
│
├── dataset/                     # Raw datasets (gitignored - large files)
│   ├── climate_dataset/        # CHIRPS & TerraClimate rasters
│   ├── elevation_dataset/      # GMTED2010 elevation data
│   ├── fire_dataset/           # FIRMS fire points (CSV)
│   ├── land_cover_dataset/     # FAO land cover shapefiles
│   └── soil_dataset/           # HWSD v2.0 soil data
│
├── models/                      # Custom ML implementations
│   ├── __init__.py
│   ├── knn.py                  # K-Nearest Neighbors classifier
│   ├── decision_tree.py        # Decision Tree (CART algorithm)
│   ├── random_forest.py        # Random Forest ensemble
│   └── README.md               # Model documentation
│
├── results/                     # Output datasets & reports
│   ├── fires.csv               # Base fire points (cleaned)
│   ├── fires_with_climate.csv  # Fire + climate features
│   ├── fires_with_elevation_features.csv
│   ├── fires_with_soil_features.csv
│   ├── fires_merged_all_features.csv  # Final merged dataset
│   └── *.txt                   # Processing summaries
│
├── scripts/                     # Utility scripts
│   ├── compare_files.py        # Dataset comparison tool
│   ├── download_data.py        # Dataset download helper
│   ├── merge_datasets.py       # Coordinate-based merging script
│   └── verify_data.py          # Dataset verification tool
│
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Conda or virtualenv
- Required packages (see `requirements.txt`)
- ~5 GB free disk space for datasets

### Installation

Follow these steps in order to set up the project:

#### 1. **Clone the repository**
```bash
git clone https://github.com/BESSAI-Tayeb/forest-fires-algeria-tunisia-analysis.git
cd forest-fires-algeria-tunisia-analysis
```

#### 2. **Create and activate conda environment**
```bash
conda create -n fire_prediction python=3.11
conda activate fire_prediction
```

#### 3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### 4. **Download the datasets**

The datasets are hosted on Google Drive and need to be downloaded separately:

```bash
python scripts/download_data.py
```

This script will:
- Provide the Google Drive link to download all datasets
- Show instructions for manual download (recommended)
- Display the expected directory structure
- List alternative data sources

**Manual Download Steps:**
1. Visit: https://drive.google.com/drive/u/5/folders/1MNp9nhtKYCA66oLxesowKBpbf_ltBn9U
2. Download all dataset folders (climate_dataset, elevation_dataset, fire_dataset, land_cover_dataset, soil_dataset)
3. Extract them to the `dataset/` directory in your project root

#### 5. **Verify the dataset installation**

After downloading, verify that all required files are present:

```bash
python scripts/verify_data.py
```

This will check:
- All dataset directories exist
- Required files are present
- File counts and sizes
- Overall dataset integrity

✅ Once verification passes, you're ready to run the analysis notebooks!

### 📦 Required Python Packages
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `geopandas` - Geospatial data processing
- `rasterio` - Raster data I/O
- `matplotlib` - Visualization
- `seaborn` - Statistical plotting
- `jupyter` - Interactive notebooks
- `scikit-learn` - Data preprocessing (StandardScaler only)
- `gdown` - Google Drive downloader (optional, for automated downloads)

## 📓 Usage

### 1. Data Processing Pipeline

**Important**: Make sure you've completed all installation steps above, including downloading and verifying the datasets!

Run the notebooks in the following order:

```bash
# Step 1: Process individual datasets
jupyter notebook analysis/elevation.ipynb    # Extract elevation features
jupyter notebook analysis/soil.ipynb         # Extract soil properties
jupyter notebook analysis/climate.ipynb      # Process climate data
jupyter notebook analysis/fire.ipynb         # Clean fire dataset

# Step 2: Merge all features
python scripts/merge_datasets.py
```

### 2. Exploratory Data Analysis

Each notebook contains comprehensive EDA sections:
- **Univariate Analysis**: Distribution plots, histograms, box plots
- **Bivariate Analysis**: Correlation matrices, pairplots, scatter plots
- **Temporal Analysis**: Seasonal patterns (climate notebook)
- **Spatial Analysis**: Geographic distribution of fire points

### 3. Machine Learning Models

Use the custom implementations in `models/`:

```python
from models.knn import KNNClassifier
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
import pandas as pd

# Load data
df = pd.read_csv('results/fires_merged_all_features.csv')
X = df.drop(['longitude', 'latitude', 'class'], axis=1).values
y = df['class'].values

# Train KNN
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)

# Train Decision Tree
dt = DecisionTree(max_depth=10, min_samples_split=5)
dt.fit(X_train, y_train)

# Train Random Forest
rf = RandomForest(n_trees=100, max_depth=15)
rf.fit(X_train, y_train)
```

See `models/README.md` for detailed documentation.

## 🔍 Key Features

### Data Processing
- **Raster Clipping**: Climate and elevation data clipped to Algeria/Tunisia boundaries
- **Coordinate-based Merging**: Robust merging on (longitude, latitude) keys
- **Missing Value Handling**: NaN values in climate data handled with zero-filling
- **Sea Point Removal**: Maritime locations filtered from fire dataset

### Custom ML Models
- **No sklearn dependency** for core algorithms (only StandardScaler for preprocessing)
- **Full implementations** with fit/predict/score API
- **Ensemble methods** (Random Forest with bootstrap sampling)
- **Probability predictions** available (predict_proba)

### Diagnostic Tools
- **Row count verification** across all datasets
- **Coordinate alignment checks** to ensure data integrity
- **Missing value diagnostics** with detailed reports
- **Processing summaries** saved to results/

## 📈 Results

The final merged dataset (`fires_merged_all_features.csv`) contains:
- ✅ **5,978 samples** (all fire points preserved)
- ✅ **42 features** (environmental + spatial)
- ✅ **Zero missing values** (complete data for all features)
- ✅ **Balanced spatial coverage** across Algeria and Tunisia

### Data Quality Metrics
- Climate data: 6.36% of points had NaN values (handled with zero-filling)
- Elevation data: 100% coverage with derived features (slope, aspect, roughness)
- Soil data: Complete coverage with 23 properties per location
- All datasets verified for coordinate alignment

## 🤝 Contributing

This is an academic project for the Data Mining course (2025-2026). Contributions and suggestions are welcome!

## 📝 License

This project is part of academic coursework and is provided for educational purposes.

## 👥 Authors

Data Mining Project - 2025/2026

## 📚 References

### Data Sources
1. **FIRMS**: https://firms.modaps.eosdis.nasa.gov/country/
2. **FAO Land Cover (Algeria)**: https://data.apps.fao.org/catalog/iso/0e958049-2a0a-4935-83c8-af78626068fc
3. **FAO Land Cover (Tunisia)**: https://data.apps.fao.org/catalog/iso/d0ba96c7-786c-4f3f-bbd9-e427b3b23d2d
4. **WorldClim Climate**: https://worldclim.org/data/monthlywth.html
5. **GMTED2010 Elevation**: https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/Grid_ZipFiles/be15_grd.zip
6. **HWSD v2.0 Soil**: https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v20/en/

### Tools & Libraries
- GDAL/Rasterio for geospatial data processing
- Pandas/NumPy for data manipulation
- GeoPandas for vector data operations
- Matplotlib/Seaborn for visualization

## 🐛 Known Issues

- Large raster files (climate, elevation, soil) are not included in the repository (see `.gitignore`)
- Climate data extraction may show NaN values for points outside raster bounds (handled with zero-filling)
- Land cover analysis notebook is separate from the main merging pipeline

## 🔮 Future Work

- [ ] Integrate land cover features into the merged dataset
- [ ] Implement additional ML models (SVM, Gradient Boosting)
- [ ] Add cross-validation framework
- [ ] Create automated model comparison pipeline
- [ ] Add feature importance analysis
- [ ] Develop fire risk mapping visualization
#
