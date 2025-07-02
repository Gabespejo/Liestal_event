#!/usr/bin/env -S mamba run -n env_py311 python
import os
import sys
import argparse
import geopandas as gpd

# make sure your src is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from DEM_processing import (
    geopackage_to_raster,
    clip_raster_to_bbox,
    resample_raster,
    convert_tif_to_asc,
    rename_file_extension,
)

# your Manning-n lookup
MANNING = {
    11: 0.033, 12: 0.200, 13: 0.200, 14: 0.100, 15: 0.100, 16: 0.100, 17: 0.100,
    21: 0.160, 31: 0.160, 32: 0.259, 33: 0.160, 34: 0.160, 35: 0.100,
    41: 0.200, 42: 0.200, 43: 0.200, 44: 0.200, 45: 0.200, 46: 0.200, 47: 0.100,
    51: 0.040, 52: 0.120, 53: 0.120, 61: 0.030, 62: 0.025, 63: 0.060, 64: 0.060,
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--areal-gpkg",  required=True)
    p.add_argument("--code-field",  default="LC_27")
    p.add_argument("--work-folder", required=True)
    p.add_argument("--output-n",    required=True)
    p.add_argument("--res-2m",      type=float, default=2.0)
    args = p.parse_args()

    os.makedirs(args.work_folder, exist_ok=True)

    # 🔹 Auto-load bounding box from text file written by 01_prepare_dem.py
    bounds_file = os.path.join(args.work_folder, "Liestal_2m_4km_bounds.txt")
    if not os.path.exists(bounds_file):
        raise FileNotFoundError(f"❌ Bounds file not found: {bounds_file}")
    with open(bounds_file, "r") as f:
        bbox_str = f.read().strip()
        xmin, ymin, xmax, ymax = map(float, bbox_str.split(","))
        print(f"✅ Loaded bounding box from {bounds_file}: {xmin}, {ymin}, {xmax}, {ymax}")

    # 1) Load GeoPackage & map to Manning values
    gdf = gpd.read_file(args.areal_gpkg)
    target_field = args.code_field + "_manning"
    gdf[target_field] = gdf[args.code_field].map(MANNING)

    # 2) Rasterize at 100 m
    tif100 = os.path.join(args.work_folder, "areal_100m.tif")
    geopackage_to_raster(gdf, target_field, tif100, resolution=100)

    # 3) Clip raster
    clipped = os.path.join(args.work_folder, "areal_clip.tif")
    clip_raster_to_bbox(
        input_raster = tif100,
        output_raster = clipped,
        bbox = (xmin, ymin, xmax, ymax),
        bbox_crs = "EPSG:2056"
    )

    # 4) Resample to 2 m
    tif2m = os.path.join(args.work_folder, "Liestal_2m.tif")
    resample_raster(
        input_raster = clipped,
        output_raster = tif2m,
        target_resolution = args.res_2m,
        resampling_method = "nearest"
    )

    # 5) Convert to ASCII and rename to .n
    asc = tif2m.replace(".tif", ".asc")
    convert_tif_to_asc(tif2m, asc, desired_nodata_value=-9999)
    rename_file_extension(asc, ".n")

    # 6) Clean up temporary files
    for tmp in [asc, tif100, clipped, tif2m]:
        if os.path.exists(tmp):
            os.remove(tmp)
            print(f"🧹 Deleted: {tmp}")

    print(f"✅ Manning raster saved: {args.output_n}")

if __name__ == "__main__":
    main()

    

