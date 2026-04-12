import os
import json
import random
import shutil
from pathlib import Path

# -------- CONFIG --------
DATA_DIR = Path('./new_data')
OUTPUT_DIR = Path('./dataset_split')
SEED = 42
TEST_PERCENT = 0.10
VAL_PERCENT = 0.10  # of remaining 90%
MASK_SUFFIX = '_mask'
# ------------------------

def main():

    images_dir = DATA_DIR / 'imgs'
    masks_dir = DATA_DIR / 'masks'

    image_ids = [
        f.stem.replace(MASK_SUFFIX, '')
        for f in images_dir.glob('*.tif')
        if MASK_SUFFIX not in f.stem
    ]

    image_ids = sorted(image_ids)
    random.seed(SEED)
    random.shuffle(image_ids)

    total = len(image_ids)

    n_test = int(total * TEST_PERCENT)
    n_remaining = total - n_test
    n_val = int(n_remaining * VAL_PERCENT)
    n_train = n_remaining - n_val

    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train+n_val]
    test_ids = image_ids[n_train+n_val:]

    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_dict = {
        "seed": SEED,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    with open(OUTPUT_DIR / 'split.json', 'w') as f:
        json.dump(split_dict, f, indent=4)

    # Create folders
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'imgs').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'masks').mkdir(parents=True, exist_ok=True)

    # Copy files
    for split_name, ids in [('train', train_ids),
                            ('val', val_ids),
                            ('test', test_ids)]:

        for id_ in ids:
            shutil.copy(images_dir / f"{id_}.tif",
                        OUTPUT_DIR / split_name / 'imgs' / f"{id_}.tif")

            shutil.copy(masks_dir / f"{id_}{MASK_SUFFIX}.tif",
                        OUTPUT_DIR / split_name / 'masks' / f"{id_}{MASK_SUFFIX}.tif")

    print("Split complete.")

if __name__ == "__main__":
    main()
