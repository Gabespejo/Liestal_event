#!/usr/bin/env -S mamba run -n env_py311 python
import os
import sys
import argparse
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from glob import glob

# make sure your src is on PYTHONPATH (use this file's folder)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.normpath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

from DEM_processing import (
    geopackage_to_raster,
    clip_raster_to_bbox,
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
    p.add_argument("--work-folder", required=True, help="Folder that contains the *bounds.txt and where temps are written")
    p.add_argument("--output-n",    required=True, help="Final output .n path")
    p.add_argument("--input-dem",   required=True, help="DEM GeoTIFF used for alignment")
    p.add_argument("--bounds-name", default=None,
                   help="Filename (not a path) of the bounds text under --work-folder "
                        "(e.g. 'Zell_2m_bounds.txt'). If omitted, will try *_bounds.txt, "
                        "then fall back to 'Liestal_2m_bounds.txt'.")
    args = p.parse_args()

    os.makedirs(args.work_folder, exist_ok=True)

    # -------- Resolve bounds file --------
    bounds_file = None
    if args.bounds_name:
        candidate = os.path.join(args.work_folder, args.bounds_name)
        if os.path.exists(candidate):
            bounds_file = candidate
        else:
            raise FileNotFoundError(f"Bounds file not found: {candidate}")
    else:
        # try any *_bounds.txt in work-folder
        candidates = sorted(glob(os.path.join(args.work_folder, "*_bounds.txt")))
        if candidates:
            bounds_file = candidates[0]
        else:
            # final fallback
            fallback = os.path.join(args.work_folder, "Liestal_2m_bounds.txt")
            if os.path.exists(fallback):
                bounds_file = fallback
            else:
                raise FileNotFoundError(
                    f"No bounds file found. Tried *_bounds.txt and {fallback} under {args.work_folder}"
                )

    with open(bounds_file, "r") as f:
        bbox_str = f.read().strip()
        xmin, ymin, xmax, ymax = map(float, bbox_str.split(","))
        print(f" Loaded bounding box from {bounds_file}: {xmin}, {ymin}, {xmax}, {ymax}")

    # 1) Load GeoPackage & map to Manning values
    gdf = gpd.read_file(args.areal_gpkg)
    target_field = args.code_field + "_manning"
    gdf[target_field] = gdf[args.code_field].map(MANNING)

    # 2) Rasterize at 100 m
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

    # 4) Align to DEM grid
    tif2m = os.path.join(args.work_folder, "manning_aligned.tif")
    with rasterio.open(args.input_dem) as src_dem:
        dem_meta = src_dem.meta.copy()
        dem_shape = src_dem.shape
        dem_transform = src_dem.transform
        dem_crs = src_dem.crs

    with rasterio.open(clipped) as src_clip:
        src_data = src_clip.read(1)

        aligned_data = np.full(dem_shape, -9999, dtype=np.float32)

        reproject(
            source=src_data,
            destination=aligned_data,
            src_transform=src_clip.transform,
            src_crs=src_clip.crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.nearest,
        )

    with rasterio.open(tif2m, "w", **dem_meta) as dst:
        dst.write(aligned_data, 1)

    # 5) Convert to ASCII and rename to .n
    asc = tif2m.replace(".tif", ".asc")
    convert_tif_to_asc(tif2m, asc, desired_nodata_value=-9999)
    rename_file_extension(asc, ".n")

    tmp_n = asc.replace(".asc", ".n")
    if os.path.abspath(tmp_n) != os.path.abspath(args.output_n):
        os.replace(tmp_n, args.output_n)

    # 6) Clean up temporary files
    for tmp in [tif100, clipped, tif2m]:
        if os.path.exists(tmp):
            os.remove(tmp)
            print(f" Deleted: {tmp}")

    print(f"✔ Manning raster aligned to DEM and saved: {args.output_n}")

if __name__ == "__main__":
    main()