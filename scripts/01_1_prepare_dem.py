#!/usr/bin/env -S mamba run -n env_py311 python
import os
import sys
import argparse

# make sure your src/ folder is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from DEM_processing import (
    resnap_dem_to_combiprecip,  # <- already in your DEM_processing
    convert_tif_to_asc,
    rename_file_extension
)

def main():
    parser = argparse.ArgumentParser(
        description="1) resnap DEM to 1 km grid, 2) convert to ASCII, 3) rename to .dem, 4) save bounds"
    )
    parser.add_argument("--input-tif", required=True,
                        help="Path to input DEM GeoTIFF (.tif)")
    parser.add_argument("--output-dem", required=True,
                        help="Path to final .dem file")
    parser.add_argument("--snap-res", type=int, default=1000,
                        help="Resolution to snap bounds (default: 1000 m for Combiprecip)")
    args = parser.parse_args()

    # --- Step 1: Resnap DEM to 1 km Combiprecip grid ---
    snapped_tif = args.output_dem.replace(".dem", "_snapped.tif")
    snapped_bounds = resnap_dem_to_combiprecip(
        input_dem=args.input_tif,
        output_dem=snapped_tif,
        snap_res=args.snap_res
    )

    # --- Step 2: Convert to Esri ASCII (.asc) ---
    asc_file = args.output_dem.replace(".dem", ".asc")
    convert_tif_to_asc(
        dem_tif=snapped_tif,
        output_asc=asc_file,
        desired_nodata_value=-9999
    )

    # --- Step 3: Rename .asc → .dem ---
    rename_file_extension(
        input_file_path=asc_file,
        new_extension=".dem"
    )

    # --- Step 4: Write bounds.txt ---
    bounds_path = args.output_dem.replace(".dem", "_bounds.txt")
    with open(bounds_path, "w") as f:
        f.write(",".join(str(b) for b in snapped_bounds))
    print(f"✔ Saved DEM bounds to {bounds_path}")

if __name__ == "__main__":
    main()