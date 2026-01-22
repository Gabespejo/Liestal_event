#!/usr/bin/env -S mamba run -n env_py311 python

import sys
import rasterio
import numpy as np
from pathlib import Path


def create_startfile(dem_path, start_path):
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile
        nodata = src.nodata

        # Zero water depth everywhere
        start = np.zeros(dem.shape, dtype=np.float32)

        # Preserve NODATA outside domain
        if nodata is not None:
            start[dem == nodata] = nodata

        profile.update(
            driver="AAIGrid",
            dtype="float32",
            nodata=nodata,
            count=1
        )

        with rasterio.open(start_path, "w", **profile) as dst:
            dst.write(start, 1)

    print("Startfile created successfully")
    print("DEM:   ", dem_path)
    print("START: ", start_path)
    print("Grid:  ", start.shape)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("  create_startfile.py <input_dem> <output_startfile>")
        sys.exit(1)

    dem_path = Path(sys.argv[1])
    start_path = Path(sys.argv[2])

    if not dem_path.exists():
        print(f"ERROR: DEM not found: {dem_path}")
        sys.exit(2)

    create_startfile(dem_path, start_path)
