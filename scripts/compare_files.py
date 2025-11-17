import pandas as pd

# Load both files
fires = pd.read_csv('results/fires_with_soil_features.csv')
fires_climate = pd.read_csv('results/fires_with_climate.csv')

print("="*80)
print("COMPARISON REPORT: fires_with_soil_features.csv vs fires_with_climate.csv")
print("="*80)

# Row counts
print(f"\nRow Counts:")
print(f"  fires_with_soil_features.csv: {len(fires)} rows")
print(f"  fires_with_climate.csv: {len(fires_climate)} rows")

# Column comparison
print(f"\nColumns:")
print(f"  fires_with_soil_features.csv: {list(fires.columns)}")
print(f"  fires_with_climate.csv: {list(fires_climate.columns)}")

# Check if coordinates match row-by-row
lon_match = (fires['longitude'].values == fires_climate['longitude'].values).all()
lat_match = (fires['latitude'].values == fires_climate['latitude'].values).all()
class_match = (fires['class'].values == fires_climate['class'].values).all()

print(f"\nRow-by-Row Matching:")
print(f"  Longitude match: {lon_match}")
print(f"  Latitude match: {lat_match}")
print(f"  Class match: {class_match}")

# Check for any mismatches
if not (lon_match and lat_match):
    print("\nFinding mismatches...")
    mismatches = []
    for i in range(min(len(fires), len(fires_climate))):
        if fires.loc[i, 'longitude'] != fires_climate.loc[i, 'longitude'] or \
           fires.loc[i, 'latitude'] != fires_climate.loc[i, 'latitude']:
            mismatches.append(i)
    
    print(f"  Total mismatched rows: {len(mismatches)}")
    if len(mismatches) > 0 and len(mismatches) <= 10:
        print(f"  Mismatched row indices: {mismatches}")

# Check unique coordinates
fires['coord_key'] = fires['longitude'].astype(str) + ',' + fires['latitude'].astype(str)
fires_climate['coord_key'] = fires_climate['longitude'].astype(str) + ',' + fires_climate['latitude'].astype(str)

unique_fires = set(fires['coord_key'])
unique_climate = set(fires_climate['coord_key'])

missing = unique_fires - unique_climate
extra = unique_climate - unique_fires

print(f"\nSet Comparison:")
print(f"  Unique coordinates in fires.csv: {len(unique_fires)}")
print(f"  Unique coordinates in fires_with_climate.csv: {len(unique_climate)}")
print(f"  Missing from fires_with_climate.csv: {len(missing)}")
print(f"  Extra in fires_with_climate.csv: {len(extra)}")

# Final verdict
print("\n" + "="*80)
if len(fires) == len(fires_climate) and lon_match and lat_match and class_match:
    print("✅ PERFECT MATCH!")
    print("Both files have identical points (longitude, latitude, class) in the same order.")
elif len(fires) == len(fires_climate) and len(missing) == 0 and len(extra) == 0:
    print("⚠️ PARTIAL MATCH")
    print("Same points exist but in different order.")
else:
    print("❌ MISMATCH")
    print("Files have different points or different row counts.")
print("="*80)
