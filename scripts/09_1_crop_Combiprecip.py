#!/usr/bin/env -S mamba run -n env_py311 python
import sys
import argparse
import os
import tempfile
import numpy as np
import xarray as xr

# Make sure your src/ is on the path
sys.path.insert(0, "/storage/homefs/ge24z347/Zell_event/src/")

from lisflood_inputdata import crop_deterministic_Combiprecip


def _parse_times_csv(s: str):
    return [t.strip() for t in s.split(",") if t.strip()]


def _prepare_input_nc(orig_nc, time_name="time"):
    """
    Return a NetCDF path safe for xarray .sel(...).
    If time is unique, return the original file path.
    If time has duplicates, create a temporary cleaned file and return that path.
    """
    ds = xr.open_dataset(orig_nc)

    if time_name not in ds.indexes:
        ds.close()
        raise KeyError(f"'{time_name}' is not an index in the dataset.")

    is_unique = ds.indexes[time_name].is_unique
    print(f"{time_name} unique: {is_unique}")

    if is_unique:
        ds.close()
        return orig_nc, None

    print(f"Duplicate values found in '{time_name}'. Removing duplicates and keeping first occurrence.")

    _, unique_idx = np.unique(ds[time_name].values, return_index=True)
    unique_idx = np.sort(unique_idx)
    ds_clean = ds.isel({time_name: unique_idx})

    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tmp_path = tmp.name
    tmp.close()

    ds_clean.to_netcdf(tmp_path)
    ds.close()
    ds_clean.close()

    print(f"Temporary cleaned file created: {tmp_path}")
    return tmp_path, tmp_path


def main():
    p = argparse.ArgumentParser(
        description="Crop Combiprecip NetCDF to a DEM footprint and selected timestamps."
    )
    p.add_argument("--orig-nc", required=True, help="Input CPC NetCDF")
    p.add_argument("--dem-file", required=True, help="DEM file to get bbox from")
    p.add_argument("--output-nc", required=True, help="Output cropped NetCDF")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--times", help="Comma-separated ISO times")
    g.add_argument("--times-file", help="Text file with one ISO time per line")

    args = p.parse_args()

    if args.times:
        selected_times = _parse_times_csv(args.times)
    else:
        with open(args.times_file) as f:
            selected_times = [ln.strip() for ln in f if ln.strip()]

    input_nc_for_crop, temp_file_to_delete = _prepare_input_nc(args.orig_nc, time_name="time")

    try:
        crop_deterministic_Combiprecip(
            orig_nc=input_nc_for_crop,
            dem_file=args.dem_file,
            output_nc=args.output_nc,
            selected_times=selected_times,
            var_name="RR",
            time_name="time",
            x_name="x",
            y_name="y",
        )
    finally:
        if temp_file_to_delete is not None and os.path.exists(temp_file_to_delete):
            os.remove(temp_file_to_delete)
            print(f"Removed temporary file: {temp_file_to_delete}")


if __name__ == "__main__":
    main()