"""
Download Dataset Script for Fire Risk Prediction Project

This script automates the dataset download and setup process.
The dataset includes climate, elevation, soil, fire, and land cover data.

Google Drive Link: https://drive.google.com/drive/u/5/folders/1MNp9nhtKYCA66oLxesowKBpbf_ltBn9U

Workflow:
    1. Download dataset.zip from Google Drive
    2. Extract contents to dataset/ directory
    3. Run automatic verification via verify_data.py

Requirements:
    pip install gdown

Configuration:
    Before running, you must configure the file ID:
    1. Open the Google Drive folder link above
    2. Right-click on 'dataset.zip' -> Get link
    3. Extract the file ID from the URL
    4. Replace 'YOUR_FILE_ID_HERE' in the download_dataset() function

Usage:
    python scripts/download_data.py
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import gdown
        print("✓ gdown is installed")
        return True
    except ImportError:
        print("❌ gdown is not installed")
        print("\nPlease install gdown using:")
        print("  pip install gdown")
        return False

def download_dataset():
    """Download the dataset from Google Drive."""
    
    file_id = "1jzSDxkb3csbDWQ3rWV2w6U7BvgNitqm_"
    folder_url = "https://drive.google.com/drive/folders/1U8Pwz_UhZezCpqvX6RZl8uwSoxtHkQoJ"
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "dataset"
    download_path = project_root / "dataset.zip"
    
    print("="*80)
    print("FIRE RISK DATASET DOWNLOADER")
    print("="*80)
    print(f"\nGoogle Drive Folder: {folder_url}")
    print(f"Download Location: {dataset_dir.absolute()}")
    print(f"Zip File: {download_path.absolute()}")
    
    # Create dataset directory if it doesn't exist
    dataset_dir.mkdir(exist_ok=True)
    
    # Check if dataset.zip already exists
    if download_path.exists():
        print(f"\n✓ dataset.zip already exists")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download...")
            return True, download_path
    
    print("\n" + "="*80)
    print("DOWNLOADING DATASET")
    print("="*80)
    
    # Import gdown
    try:
        import gdown
    except ImportError:
        print("❌ Error: gdown is not installed")
        print("Please install it using: pip install gdown")
        return False, None
    
    # Download the file
    print(f"\n📥 Downloading dataset.zip from Google Drive...")
    print(f"   File ID: {file_id}")
    
    if file_id == "YOUR_FILE_ID_HERE":
        print("\n❌ ERROR: File ID not configured!")
        print("\nTo configure the file ID:")
        print("1. Go to Google Drive folder:")
        print(f"   {folder_url}")
        print("2. Right-click on 'dataset.zip' -> Get link")
        print("3. Extract the file ID from the URL")
        print(f"4. Edit this script and replace 'YOUR_FILE_ID_HERE' with the actual ID")
        print("\nAlternatively, you can download manually and place dataset.zip in:")
        print(f"   {project_root.absolute()}")
        return False, None
    
    try:
        # Download file from Google Drive
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, str(download_path), quiet=False)
        
        if download_path.exists():
            file_size = download_path.stat().st_size / (1024 * 1024)  # Convert to MB
            print(f"\n✅ Download successful! ({file_size:.2f} MB)")
            return True, download_path
        else:
            print("\n❌ Download failed: File not found after download")
            return False, None
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nYou can download manually:")
        print(f"1. Visit: {folder_url}")
        print("2. Download 'dataset.zip'")
        print(f"3. Place it in: {project_root.absolute()}")
        return False, None

def extract_dataset(zip_path):
    """Extract the downloaded zip file."""
    
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "dataset"
    
    print("\n" + "="*80)
    print("EXTRACTING DATASET")
    print("="*80)
    
    if not zip_path.exists():
        print(f"\n❌ Error: {zip_path} not found")
        return False
    
    print(f"\n📦 Extracting {zip_path.name}...")
    print(f"   Destination: {dataset_dir.absolute()}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            print(f"   Total files to extract: {total_files}")
            
            # Extract to a temporary location first
            temp_dir = project_root / "temp_extract"
            temp_dir.mkdir(exist_ok=True)
            zip_ref.extractall(temp_dir)
            
            # Check if there's a nested dataset folder
            nested_dataset = temp_dir / "dataset"
            if nested_dataset.exists() and nested_dataset.is_dir():
                print("   Detected nested 'dataset' folder, fixing structure...")
                # Move contents from nested dataset folder directly to dataset_dir
                import shutil
                for item in nested_dataset.iterdir():
                    dest = dataset_dir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dataset_dir))
                # Clean up temp directory
                shutil.rmtree(temp_dir)
            else:
                # No nested folder, move everything from temp to dataset
                import shutil
                for item in temp_dir.iterdir():
                    dest = dataset_dir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dataset_dir))
                # Clean up temp directory
                shutil.rmtree(temp_dir)
            
        print(f"\n✅ Extraction successful!")
        print(f"   {total_files} files extracted to {dataset_dir.absolute()}")
        
        # Optional: Remove zip file after extraction
        response = input("\nDo you want to delete dataset.zip to save space? (y/N): ").strip().lower()
        if response == 'y':
            zip_path.unlink()
            print("✅ dataset.zip deleted")
        
        return True
        
    except zipfile.BadZipFile:
        print("\n❌ Error: Invalid or corrupted zip file")
        return False
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        return False

def run_verification():
    """Run the verification script."""
    
    print("\n" + "="*80)
    print("RUNNING DATASET VERIFICATION")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    verify_script = project_root / "scripts" / "verify_data.py"
    
    if not verify_script.exists():
        print("\n⚠️  verify_data.py not found")
        return False
    
    print("\n🔍 Verifying dataset structure and contents...")
    
    try:
        # Run verification script
        result = subprocess.run(
            [sys.executable, str(verify_script)],
            cwd=project_root,
            capture_output=False,
            text=True
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False

def main():
    """Main function."""
    print("\nChecking system requirements...")
    
    if not check_dependencies():
        print("\n❌ Missing required dependencies. Please install them first:")
        print("   pip install gdown")
        sys.exit(1)
    
    print("✅ All dependencies available\n")
    
    # Step 1: Download dataset.zip
    success, zip_path = download_dataset()
    
    if not success:
        print("\n❌ Download failed. Please download manually.")
        sys.exit(1)
    
    # Step 2: Extract the zip file
    if not extract_dataset(zip_path):
        print("\n❌ Extraction failed.")
        sys.exit(1)
    
    # Step 3: Run verification
    if run_verification():
        print("\n" + "="*80)
        print("✅ SETUP COMPLETE!")
        print("="*80)
        print("\nYour dataset is ready for analysis.")
        print("You can now run the Jupyter notebooks in the analysis/ directory.")
    else:
        print("\n⚠️  Verification had some issues. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
