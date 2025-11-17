import pandas as pd
import os

print("="*80)
print("MERGING FIRE DATASETS")
print("="*80)

# Define file paths
results_dir = 'results'
fires_csv = os.path.join(results_dir, 'fires.csv')
climate_csv = os.path.join(results_dir, 'fires_with_climate.csv')
elevation_csv = os.path.join(results_dir, 'fires_with_elevation_features.csv')
soil_csv = os.path.join(results_dir, 'fires_with_soil_features.csv')
output_csv = os.path.join(results_dir, 'fires_merged_all_features.csv')

# Load all datasets
print("\n📂 Loading datasets...")
fires_df = pd.read_csv(fires_csv)
climate_df = pd.read_csv(climate_csv)
elevation_df = pd.read_csv(elevation_csv)
soil_df = pd.read_csv(soil_csv)

print(f"  fires.csv: {len(fires_df)} rows, {len(fires_df.columns)} columns")
print(f"  fires_with_climate.csv: {len(climate_df)} rows, {len(climate_df.columns)} columns")
print(f"  fires_with_elevation_features.csv: {len(elevation_df)} rows, {len(elevation_df.columns)} columns")
print(f"  fires_with_soil_features.csv: {len(soil_df)} rows, {len(soil_df.columns)} columns")

# Verify all datasets have the same number of rows
print("\n🔍 Verifying row counts...")
if len(fires_df) == len(climate_df) == len(elevation_df) == len(soil_df):
    print(f"  ✅ All datasets have {len(fires_df)} rows")
else:
    print("  ❌ WARNING: Datasets have different row counts!")
    print(f"     fires: {len(fires_df)}, climate: {len(climate_df)}, elevation: {len(elevation_df)}, soil: {len(soil_df)}")

# Verify coordinates match (row-by-row)
print("\n🔍 Verifying coordinate alignment...")
climate_match = (fires_df[['longitude', 'latitude']].values == climate_df[['longitude', 'latitude']].values).all()
elevation_match = (fires_df[['longitude', 'latitude']].values == elevation_df[['longitude', 'latitude']].values).all()
soil_match = (fires_df[['longitude', 'latitude']].values == soil_df[['longitude', 'latitude']].values).all()

print(f"  Climate coordinates match: {climate_match}")
print(f"  Elevation coordinates match: {elevation_match}")
print(f"  Soil coordinates match: {soil_match}")

# Merge strategy: Use pandas merge on longitude and latitude
print("\n🔧 Merging datasets using coordinate-based merge...")

# Start with fires base data
merged_df = fires_df.copy()

# Merge climate features
print("\n  Step 1: Merging climate features...")
climate_cols_to_merge = [col for col in climate_df.columns if col not in ['longitude', 'latitude', 'class']]
merged_df = merged_df.merge(
    climate_df[['longitude', 'latitude'] + climate_cols_to_merge],
    on=['longitude', 'latitude'],
    how='left'
)
print(f"     Added {len(climate_cols_to_merge)} climate columns: {climate_cols_to_merge}")
print(f"     Rows after merge: {len(merged_df)}")

# Merge elevation features
print("\n  Step 2: Merging elevation features...")
elevation_cols_to_merge = [col for col in elevation_df.columns if col not in ['longitude', 'latitude', 'class']]
merged_df = merged_df.merge(
    elevation_df[['longitude', 'latitude'] + elevation_cols_to_merge],
    on=['longitude', 'latitude'],
    how='left'
)
print(f"     Added {len(elevation_cols_to_merge)} elevation columns: {elevation_cols_to_merge}")
print(f"     Rows after merge: {len(merged_df)}")

# Merge soil features
print("\n  Step 3: Merging soil features...")
soil_cols_to_merge = [col for col in soil_df.columns if col not in ['longitude', 'latitude', 'class']]
merged_df = merged_df.merge(
    soil_df[['longitude', 'latitude'] + soil_cols_to_merge],
    on=['longitude', 'latitude'],
    how='left'
)
print(f"     Added {len(soil_cols_to_merge)} soil columns: {soil_cols_to_merge}")
print(f"     Rows after merge: {len(merged_df)}")

if True:
    
    # Summary
    print(f"\n📊 Merged Dataset Summary:")
    print(f"  Total rows: {len(merged_df)}")
    print(f"  Total columns: {len(merged_df.columns)}")
    base_cols = ['longitude', 'latitude', 'class']
    print(f"  Base columns (3): {base_cols}")
    print(f"  Climate columns ({len(climate_cols_to_merge)}): {climate_cols_to_merge}")
    print(f"  Elevation columns ({len(elevation_cols_to_merge)}): {elevation_cols_to_merge}")
    print(f"  Soil columns ({len(soil_cols_to_merge)}): {soil_cols_to_merge}")
    
    # Check for missing values
    print(f"\n🔍 Checking for missing values...")
    missing_counts = merged_df.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"  ⚠️  Found {total_missing} missing values:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"     {col}: {count} missing ({count/len(merged_df)*100:.2f}%)")
    else:
        print(f"  ✅ No missing values found")
    
    # Save merged dataset
    print(f"\n💾 Saving merged dataset to: {output_csv}")
    merged_df.to_csv(output_csv, index=False)
    print(f"  ✅ Saved successfully!")
    
    # Display preview
    print("\n" + "="*80)
    print("MERGED DATASET PREVIEW (first 5 rows)")
    print("="*80)
    print(merged_df.head())
    
    print("\n" + "="*80)
    print("COLUMN NAMES")
    print("="*80)
    print(list(merged_df.columns))
    
    print("\n" + "="*80)
    print("✅ MERGE COMPLETE!")
    print("="*80)
    print(f"\nMerged file saved: {output_csv}")
    print(f"Total features: {len(merged_df.columns)} columns")
    print(f"Total samples: {len(merged_df)} rows")

