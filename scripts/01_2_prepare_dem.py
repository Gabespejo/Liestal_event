#!/usr/bin/env -S mamba run -n env_py311 python
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from files_preparing_to_lisflood import (
    crop_dem_to_Combiprecip_1km,
    convert_tif_to_asc,
    rename_file_extension
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "1) crop full DEM to a square domain based on a catchment polygon "
            "(snapped to 1 km grid), "
            "2) convert to ASCII, "
            "3) rename to .dem, "
            "4) save bounds"
        )
    )

    parser.add_argument("--full-dem", required=True,
                        help="Full domain DEM GeoTIFF (base DEM)")
    parser.add_argument("--catchment-shp", required=True,
                        help="Catchment polygon shapefile / gpkg")
    parser.add_argument("--id-field", required=True,
                        help="Column name identifying catchment (e.g. ID)")
    parser.add_argument("--id-value", required=True,
                        help="Value in id-field identifying catchment")
    parser.add_argument("--output-dem", required=True,
                        help="Path to final .dem file")
    parser.add_argument("--snap-res", type=int, default=1000,
                        help="Snap bounds to this grid (default: 1000 m)")
    parser.add_argument("--pad-m", type=float, default=1000.0,
                        help="Padding (meters) added around catchment (default: 1000)")
    parser.add_argument("--mode", choices=["out", "in"], default="out",
                        help="Snap mode (default: out)")
    parser.add_argument("--no-square", action="store_true",
                        help="Disable square domain (not recommended)")
    parser.add_argument("--enforce-epsg2056", action="store_true",
                        help="Require DEM CRS to be EPSG:2056")

    args = parser.parse_args()

    # Convert id_value to int/float if possible
    id_value = args.id_value
    try:
        id_value = int(id_value)
    except ValueError:
        try:
            id_value = float(id_value)
        except ValueError:
            pass

    # --- Step 1: Crop DEM ---
    cropped_tif = args.output_dem.replace(".dem", "_crop_1km.tif")

    snapped_bounds = crop_dem_to_Combiprecip_1km(
        full_dem=args.full_dem,
        catchment_shp=args.catchment_shp,
        id_field=args.id_field,
        id_value=id_value,
        output_dem=cropped_tif,
        snap_res=args.snap_res,
        mode=args.mode,
        make_square=not args.no_square,
        pad_m=args.pad_m,
        enforce_epsg2056=args.enforce_epsg2056
    )

    # --- Step 2: Convert to ASCII ---
    asc_file = args.output_dem.replace(".dem", ".asc")
    convert_tif_to_asc(
        dem_tif=cropped_tif,
        output_asc=asc_file,
        desired_nodata_value=-9999
    )

    # --- Step 3: Rename .asc → .dem ---
    rename_file_extension(
        input_file_path=asc_file,
        new_extension=".dem"
    )

    # --- Step 4: Save bounds ---
    bounds_path = args.output_dem.replace(".dem", "_bounds.txt")
    with open(bounds_path, "w") as f:
        f.write(",".join(str(b) for b in snapped_bounds))

    print("✔ DEM prepared for LISFLOOD")
    print(f"✔ Final DEM:     {args.output_dem}")
    print(f"✔ Bounds file:   {bounds_path}")
    print(f"✔ Padding used:  {args.pad_m} m")


if __name__ == "__main__":
    main()
