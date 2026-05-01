#!/usr/bin/env -S mamba run -n env_py311 python
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from DEM_processing import resnap_dem_to_combiprecip, convert_tif_to_asc, rename_file_extension

import DTM_and_DSM
print("DTM_and_DSM loaded from:", DTM_and_DSM.__file__)

from DTM_and_DSM import rasterize_buildings, DTM_DSM_both


def main():
    parser = argparse.ArgumentParser(description="Prepare DEM pipeline (snap + buildings + DSM/DTM mix + .dem + bounds)")
    parser.add_argument("--input-dtm-tif", required=True)
    parser.add_argument("--full-dem", required=True)
    parser.add_argument("--dsm-tif", required=True)
    parser.add_argument("--buildings", required=True)

    parser.add_argument("--out-folder", required=True)
    parser.add_argument("--out-name", default="Zell_2m")
    parser.add_argument("--snap-res", type=int, default=1000)

    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)

    # 1) Snap DTM
    snapped_dtm_tif = os.path.join(args.out_folder, f"{args.out_name}_snapped.tif")
    snapped_bounds = resnap_dem_to_combiprecip(
        input_dem=args.input_dtm_tif,
        output_dem=snapped_dtm_tif,
        snap_res=args.snap_res,
        mode="out",
        full_dem=args.full_dem
    )
    print(f"✔ Snapped DTM saved: {snapped_dtm_tif}")

    # 2) Buildings mask name = same as geojson but .tif (saved next to the geojson)
    buildings_mask_tif = os.path.join(args.out_folder, os.path.splitext(os.path.basename(args.buildings))[0] + ".tif")
    print("Buildings mask will be:", buildings_mask_tif)

    if not os.path.exists(args.buildings):
        raise FileNotFoundError(f"Buildings vector not found: {args.buildings}")

    rasterize_buildings(
        buildings_vector=args.buildings,
        dtm_tif=snapped_dtm_tif,
        out_buildings_mask_tif=buildings_mask_tif
    )

    # 3) Mixed DEM
    dem_mix_tif = os.path.join(args.out_folder, f"{args.out_name}.tif")
    DTM_DSM_both(snapped_dtm_tif, args.dsm_tif, buildings_mask_tif, dem_mix_tif)
    print(f"✔ Mixed DEM saved: {dem_mix_tif}")

    # 4) Convert to .dem
    asc_file = os.path.join(args.out_folder, f"{args.out_name}.asc")
    convert_tif_to_asc(dem_tif=dem_mix_tif, output_asc=asc_file, desired_nodata_value=-9999)
    rename_file_extension(input_file_path=asc_file, new_extension=".dem")

    # 5) Bounds
    bounds_path = os.path.join(args.out_folder, f"{args.out_name}_bounds.txt")
    with open(bounds_path, "w") as f:
        f.write(",".join(str(b) for b in snapped_bounds))
    print(f"✔ Saved bounds to: {bounds_path}")


if __name__ == "__main__":
    main()
