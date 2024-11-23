import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import shutil

# Define paths
datasets = {
    "Valencia": {
        "after_flood": "202410Valencia/202410Valencia_AF.tif",
        "before_flood": "202410Valencia/202410Valencia_BF.tif",
        "flood_mask": "202410Valencia/202410Valencia_FM.tif",
        "perm_water": "202410Valencia/202410Valencia_PWL.tif",
    },
    "Mississippi": {
        "after_flood": "201905Mississipi/201905Mississipi_AF.tif",
        "before_flood": "201905Mississipi/201905Mississipi_BF.tif",
        "flood_mask": "201905Mississipi/201905Mississipi_FM.tif",
        "perm_water": "201905Mississipi/201905Mississipi_PWL.tif",
    }
}

output_dir = "dataset"
tile_size = 256
flood_threshold = 0.1  # Minimum 10% flood coverage

# Ensure output directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations", split), exist_ok=True)

# Helper function to create tiles
def process_dataset(dataset_name, paths):
    print(f"Processing {dataset_name} dataset...")
    with rasterio.open(paths["after_flood"]) as af, \
         rasterio.open(paths["before_flood"]) as bf, \
         rasterio.open(paths["flood_mask"]) as fm, \
         rasterio.open(paths["perm_water"]) as pw:

        # Get metadata
        width, height = af.width, af.height

        # Iterate through tiles
        for i in tqdm(range(0, width, tile_size)):
            for j in range(0, height, tile_size):
                window = Window(i, j, tile_size, tile_size)

                # Read data
                af_tile = af.read(window=window)
                bf_tile = bf.read(window=window)
                fm_tile = fm.read(1, window=window)  # Single band
                pw_tile = pw.read(1, window=window)  # Single band

                # Check flood coverage
                flood_coverage = np.sum(fm_tile > 0) / (tile_size * tile_size)
                if flood_coverage < flood_threshold:
                    continue

                # Combine SAR data into multi-channel
                combined_tile = np.vstack([af_tile, bf_tile])

                # Save images and masks
                tile_name = f"{dataset_name}_{i}_{j}.tif"
                img_path = os.path.join(output_dir, "images", "train", tile_name)  # Adjust split as needed
                mask_path = os.path.join(output_dir, "annotations", "train", tile_name)  # Adjust split as needed

                with rasterio.open(
                    img_path,
                    "w",
                    driver="GTiff",
                    height=tile_size,
                    width=tile_size,
                    count=combined_tile.shape[0],
                    dtype=combined_tile.dtype,
                    crs=af.crs,
                    transform=af.window_transform(window),
                ) as dst:
                    dst.write(combined_tile)

                # Combine masks (flood and permanent water)
                combined_mask = np.zeros_like(fm_tile, dtype=np.uint8)
                combined_mask[fm_tile > 0] = 1  # Flood
                combined_mask[pw_tile > 0] = 2  # Permanent water

                with rasterio.open(
                    mask_path,
                    "w",
                    driver="GTiff",
                    height=tile_size,
                    width=tile_size,
                    count=1,
                    dtype=np.uint8,
                    crs=fm.crs,
                    transform=fm.window_transform(window),
                ) as dst:
                    dst.write(combined_mask, 1)

# Process each dataset
for name, paths in datasets.items():
    process_dataset(name, paths)

# Split into train/val/test (basic random split)
def split_data():
    all_tiles = os.listdir(os.path.join(output_dir, "images", "train"))
    np.random.shuffle(all_tiles)
    train, val, test = np.split(
        all_tiles, [int(0.8 * len(all_tiles)), int(0.9 * len(all_tiles))]
    )

    for split, tiles in zip(["train", "val", "test"], [train, val, test]):
        for tile in tiles:
            shutil.move(
                os.path.join(output_dir, "images", "train", tile),
                os.path.join(output_dir, "images", split, tile),
            )
            shutil.move(
                os.path.join(output_dir, "annotations", "train", tile),
                os.path.join(output_dir, "annotations", split, tile),
            )

split_data()