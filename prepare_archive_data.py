#!/usr/bin/env python3
"""
Data preparation script for brain MRI segmentation dataset from archive.zip
Handles structure: archive.zip -> patient_folders -> images and masks
"""

import os
import zipfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def unzip_archive(zip_path='./archive.zip', extract_to='./data_raw'):
    """Unzip the archive.zip file"""
    print(f"Unzipping {zip_path}...")
    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files to show progress
        file_list = zip_ref.namelist()
        print(f"Found {len(file_list)} files in archive")
        
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, extract_path)
    
    print(f"Dataset extracted to {extract_path}")
    return extract_path

def find_patient_folders(raw_data_path):
    """
    Find all patient folders containing images and masks
    Expected structure:
      archive.zip/
        patient_folder_1/
          image1.tif
          image1_mask.tif
          image2.tif
          image2_mask.tif
        patient_folder_2/
          ...
    """
    raw_path = Path(raw_data_path)
    patient_folders = []
    
    print("Scanning for patient directories with images...")
    
    # Walk through all directories
    for root, dirs, files in os.walk(raw_path):
        # Check if this directory contains .tif files
        tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]
        
        if tif_files:
            # Check if there are both images and masks
            has_images = any('mask' not in f.lower() for f in tif_files)
            has_masks = any('mask' in f.lower() for f in tif_files)
            
            if has_images and has_masks:
                patient_folders.append(Path(root))
    
    return patient_folders

def prepare_brain_mri_data(raw_data_path, output_base='./new_data'):
    """
    Organize brain MRI data into the structure expected by train.py
    
    Output structure:
    - ./data/imgs/ - all MRI images
    - ./data/masks/ - corresponding masks
    """
    raw_path = Path(raw_data_path)
    imgs_dir = Path(output_base) / 'imgs'
    masks_dir = Path(output_base) / 'masks'
    
    # Create output directories
    imgs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Find patient folders
    patient_folders = find_patient_folders(raw_path)
    
    if not patient_folders:
        print("ERROR: No patient directories with images and masks found!")
        print(f"Searched in: {raw_path}")
        print("\nDirectory structure found:")
        for root, dirs, files in os.walk(raw_path):
            level = root.replace(str(raw_path), '').count(os.sep)
            if level < 3:  # Only show first 3 levels
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                if level < 2:
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:
                        print(f'{subindent}{file}')
                    if len(files) > 5:
                        print(f'{subindent}... and {len(files) - 5} more files')
        return False
    
    print(f"Found {len(patient_folders)} patient directories with images")
    
    total_images = 0
    total_masks = 0
    skipped = 0
    
    # Process each patient directory
    for patient_dir in tqdm(patient_folders, desc="Processing patients"):
        patient_id = patient_dir.name
        
        # Get all .tif files in this directory
        tif_files = list(patient_dir.glob('*.tif')) + list(patient_dir.glob('*.tiff'))
        
        # Separate images and masks
        image_files = [f for f in tif_files if 'mask' not in f.name.lower()]
        mask_files = [f for f in tif_files if 'mask' in f.name.lower()]
        
        # Process each image and its corresponding mask
        for img_file in image_files:
            # Find corresponding mask
            # Mask filename is typically: imagename_mask.tif
            base_name = img_file.stem  # filename without extension
            
            # Look for mask with same base name
            corresponding_mask = None

            expected_mask_name = f"{base_name}_mask"
            
            for mask_file in mask_files:
                if mask_file.stem == expected_mask_name:
                    corresponding_mask = mask_file
                    break

            
            if corresponding_mask is None:
                print(f"Warning: No mask found for {img_file.name}, skipping...")
                skipped += 1
                continue
            
            # Create unique filenames using patient_id and original filename
            new_img_name = f"{patient_id}_{img_file.name}"
            new_mask_name = f"{patient_id}_{corresponding_mask.name}"
            
            # Copy files
            shutil.copy2(img_file, imgs_dir / new_img_name)
            shutil.copy2(corresponding_mask, masks_dir / new_mask_name)
            
            total_images += 1
            total_masks += 1
    
    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"{'='*60}")
    print(f"Total images: {total_images}")
    print(f"Total masks: {total_masks}")
    print(f"Skipped (no mask): {skipped}")
    print(f"Images directory: {imgs_dir}")
    print(f"Masks directory: {masks_dir}")
    
    if total_images == 0:
        print("\nERROR: No images were processed!")
        return False
    
    # Verify some images
    print("\nVerifying sample images...")
    sample_images = list(imgs_dir.glob('*.tif'))[:3]
    for img_path in sample_images:
        try:
            img = Image.open(img_path)
            print(f"  ✓ {img_path.name}: size={img.size}, mode={img.mode}")
        except Exception as e:
            print(f"  ✗ {img_path.name}: ERROR - {e}")
    
    # Verify masks have correct values
    print("\nVerifying sample masks...")
    sample_masks = list(masks_dir.glob('*.tif'))[:3]
    for mask_path in sample_masks:
        try:
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            unique_vals = np.unique(mask_array)
            print(f"  ✓ {mask_path.name}: size={mask.size}, mode={mask.mode}, values={unique_vals}")
        except Exception as e:
            print(f"  ✗ {mask_path.name}: ERROR - {e}")
    
    return True

def inspect_archive_structure(zip_path='./archive.zip'):
    """Inspect the structure of archive.zip without extracting"""
    print(f"\n{'='*60}")
    print("Archive Structure Inspection")
    print(f"{'='*60}")
    
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found!")
        return
    
    print(f"Archive: {zip_path}")
    print(f"Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB\n")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()
        print(f"Total files in archive: {len(file_list)}\n")
        
        # Group by directory
        directories = {}
        for filename in file_list:
            parts = filename.split('/')
            if len(parts) > 1:
                dir_name = parts[0] if parts[0] else parts[1]
                if dir_name not in directories:
                    directories[dir_name] = []
                directories[dir_name].append(filename)
        
        print(f"Found {len(directories)} top-level directories/folders\n")
        
        # Show first few directories
        for i, (dir_name, files) in enumerate(list(directories.items())[:5]):
            print(f"{dir_name}/ ({len(files)} files)")
            # Show first few files in this directory
            for f in files[:5]:
                print(f"  {os.path.basename(f)}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
            print()
        
        if len(directories) > 5:
            print(f"... and {len(directories) - 5} more directories")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare brain MRI dataset from archive.zip')
    parser.add_argument('--zip-path', type=str, default='./archive.zip',
                        help='Path to archive.zip file (default: ./archive.zip)')
    parser.add_argument('--output-dir', type=str, default='./new_data',
                        help='Output directory for organized data (default: ./new_data)')
    parser.add_argument('--raw-dir', type=str, default='./data_raw',
                        help='Directory to extract raw data (default: ./data_raw)')
    parser.add_argument('--inspect-only', action='store_true',
                        help='Only inspect the archive structure without processing')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip extraction if data_raw already exists')
    
    args = parser.parse_args()
    
    # Check if zip file exists
    if not os.path.exists(args.zip_path):
        print(f"ERROR: Archive file not found: {args.zip_path}")
        print("\nPlease ensure archive.zip is in the project root directory")
        exit(1)
    
    # Step 1: Inspect archive structure
    inspect_archive_structure(args.zip_path)
    
    if args.inspect_only:
        print("\nInspection complete. Use without --inspect-only to process the data.")
        exit(0)
    
    # Step 2: Extract archive
    if args.skip_extract and os.path.exists(args.raw_dir):
        print(f"\nSkipping extraction - using existing {args.raw_dir}")
        raw_data_path = Path(args.raw_dir)
    else:
        raw_data_path = unzip_archive(args.zip_path, args.raw_dir)
    
    # Step 3: Prepare and organize data
    print(f"\n{'='*60}")
    print("Organizing Data")
    print(f"{'='*60}\n")
    
    success = prepare_brain_mri_data(raw_data_path, args.output_dir)
    
    if success:
        print("\n" + "="*60)
        print("✓ Setup complete! You can now run training:")
        print("="*60)
        print("\n1. Update data loader:")
        print("   cp utils_data_loading_updated.py utils/data_loading.py")
        print("\n2. Start training:")
        print("   python train.py --epochs 50 --batch-size 4 --learning-rate 0.0001 --amp")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Data preparation failed!")
        print("="*60)
        print("\nPlease check the error messages above.")