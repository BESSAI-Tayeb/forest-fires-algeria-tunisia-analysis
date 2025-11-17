"""
Dataset Verification Script

Verifies that all required dataset files and directories are present
and provides information about the dataset structure.

Usage:
    python scripts/verify_data.py
"""

import os
from pathlib import Path

def get_dir_size(path):
    """Calculate total size of directory."""
    total_size = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total_size += entry.stat().st_size
            elif entry.is_dir():
                total_size += get_dir_size(entry.path)
    except PermissionError:
        pass
    return total_size

def format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def count_files(path, extension=None):
    """Count files in directory, optionally filter by extension."""
    count = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                if extension is None or entry.name.endswith(extension):
                    count += 1
            elif entry.is_dir():
                count += count_files(entry.path, extension)
    except (PermissionError, FileNotFoundError):
        pass
    return count

def verify_dataset():
    """Verify dataset structure and contents."""
    
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "dataset"
    
    print("="*80)
    print("DATASET VERIFICATION REPORT")
    print("="*80)
    print(f"\nDataset Location: {dataset_dir.absolute()}\n")
    
    if not dataset_dir.exists():
        print("❌ Dataset directory does not exist!")
        print(f"   Please create: {dataset_dir.absolute()}")
        return False
    
    # Define expected structure
    datasets = {
        "climate_dataset": {
            "subdirs": ["prec", "tmax", "tmin"],
            "file_types": [".tif"],
            "description": "CHIRPS precipitation & TerraClimate temperature rasters"
        },
        "elevation_dataset": {
            "subdirs": ["be15_grd", "elevation_grd"],
            "file_types": [".adf", ".xml"],
            "description": "GMTED2010 elevation data"
        },
        "fire_dataset": {
            "files": ["algeria.csv", "fires.csv", "tunisia.csv"],
            "file_types": [".csv"],
            "description": "FIRMS fire point data"
        },
        "land_cover_dataset": {
            "subdirs": ["algeria", "tunisia"],
            "file_types": [".shp", ".dbf", ".prj"],
            "description": "FAO land cover shapefiles"
        },
        "soil_dataset": {
            "files": ["HWSD2.bil", "HWSD2.hdr", "HWSD2.mdb", "HWSD2.prj", "HWSD2.stx"],
            "file_types": [".bil", ".hdr", ".mdb"],
            "description": "Harmonized World Soil Database v2.0"
        }
    }
    
    all_valid = True
    
    # Check each dataset
    for dataset_name, config in datasets.items():
        dataset_path = dataset_dir / dataset_name
        
        print("-" * 80)
        print(f"📁 {dataset_name}")
        print(f"   {config['description']}")
        print("-" * 80)
        
        if not dataset_path.exists():
            print(f"   ❌ Directory not found: {dataset_path}")
            all_valid = False
            continue
        
        print(f"   ✓ Directory exists")
        
        # Check subdirectories
        if "subdirs" in config:
            print(f"\n   Subdirectories:")
            for subdir in config["subdirs"]:
                subdir_path = dataset_path / subdir
                if subdir_path.exists():
                    file_count = count_files(subdir_path)
                    size = format_size(get_dir_size(subdir_path))
                    print(f"      ✓ {subdir}/ ({file_count} files, {size})")
                else:
                    print(f"      ✗ {subdir}/ NOT FOUND")
                    all_valid = False
        
        # Check specific files
        if "files" in config:
            print(f"\n   Required Files:")
            for filename in config["files"]:
                file_path = dataset_path / filename
                if file_path.exists():
                    size = format_size(file_path.stat().st_size)
                    print(f"      ✓ {filename} ({size})")
                else:
                    print(f"      ✗ {filename} NOT FOUND")
                    all_valid = False
        
        # Count files by type
        if "file_types" in config:
            print(f"\n   File Type Summary:")
            for ext in config["file_types"]:
                count = count_files(dataset_path, ext)
                print(f"      {ext}: {count} files")
        
        # Show total size
        total_size = format_size(get_dir_size(dataset_path))
        print(f"\n   Total Size: {total_size}")
        print()
    
    # Overall summary
    print("="*80)
    if all_valid:
        print("✅ ALL DATASETS VERIFIED SUCCESSFULLY")
        print("\nYou can now run the analysis notebooks:")
        print("  jupyter notebook analysis/elevation.ipynb")
        print("  jupyter notebook analysis/soil.ipynb")
        print("  jupyter notebook analysis/climate.ipynb")
        print("  jupyter notebook analysis/fire.ipynb")
    else:
        print("⚠️  SOME DATASETS ARE MISSING OR INCOMPLETE")
        print("\nPlease download the missing datasets using:")
        print("  python scripts/download_data.py")
    print("="*80)
    
    return all_valid

def main():
    """Main function."""
    print()
    verify_dataset()
    print()

if __name__ == "__main__":
    main()
