#!/usr/bin/env -S mamba run -n env_py311 python
import os, sys, argparse

# ─── make sure your src/ folder is on PYTHONPATH ───────────────────────────────
SRC = "/storage/homefs/ge24z347/LISFLOOD_FP_8_1/src"
sys.path.insert(0, SRC)

from lisflood_inputdata import crop_icon_to_dem

parser = argparse.ArgumentParser(
    description="Crop ICON forecast onto your DEM grid and write per-realization NetCDFs"
)
parser.add_argument("--orig-nc",          required=True,
                    help="path to original ICON .nc file")
parser.add_argument("--dem-file",         required=True,
                    help="path to your clipped DEM GeoTIFF")
parser.add_argument("--output-folder",    required=True,
                    help="where to write Liestal_2m_*.nc files")
parser.add_argument("--target-time-str",  default="2024-06-25T00:00:00",
                    help="forecast_reference_time (ISO string)")
parser.add_argument("--max-lead-hours",   type=int, default=5,
                    help="maximum lead time to keep (in hours)")
args = parser.parse_args()

crop_icon_to_dem(
    orig_nc         = args.orig_nc,
    dem_file        = args.dem_file,
    output_folder   = args.output_folder,
    target_time_str = args.target_time_str,
    max_lead_hours  = args.max_lead_hours
)

