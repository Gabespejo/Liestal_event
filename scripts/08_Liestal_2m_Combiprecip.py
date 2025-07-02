#!/usr/bin/env -S mamba run -n env_py311 python

import sys
import os

# Add src/ to Python path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from flow_depth_plotting import plot_Liestal_Combiprecip_perhour

# ─── Parameters ──────────────────────────────────────────────────────────────

dem_file = "/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Liestal_2m/Liestal_2m.dem"
geo_shapefile = "/rs_scratch/users/ge24z347/geo_ezgg_2km_ge.shp"
bounds_file = "/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Liestal_2m/Liestal_2m_4km_bounds.txt"
initial_datetime_str = "2024-06-25T15:00:00"
lead_times_hours = list(range(11))  # [0, 1, ..., 10]
color1 = "navajowhite"
color2 = "darkorange"
color3 = "firebrick"

# ─── Bounding box ────────────────────────────────────────────────────────────

with open(bounds_file, "r") as f:
    minx, miny, maxx, maxy = map(float, f.read().strip().split(","))

xlim = (minx, maxx)
ylim = (miny, maxy)

# ─── Define paths ────────────────────────────────────────────────────────────

wd_folder = "/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Liestal_2m/Liestal_2m/"
output_folder = "/storage/homefs/ge24z347/LISFLOOD_FP_8_1/LIESTAL_PLOTS/Liestal_2m_Combiprecip/"

# ─── Ensure output directory exists ──────────────────────────────────────────

os.makedirs(output_folder, exist_ok=True)

# ─── Plotting ────────────────────────────────────────────────────────────────

print(f"Plotting single scenario from {wd_folder}")

plot_Liestal_Combiprecip_perhour(
    dem_file=dem_file,
    wd_folder=wd_folder,
    plot_output_folder=output_folder,
    geo_ezgg_2km_ge=geo_shapefile,
    plot_title_prefix="Liestal",
    initial_datetime_str=initial_datetime_str,
    lead_times_hours=lead_times_hours,
    color1=color1,
    color2=color2,
    color3=color3,
    xlim=xlim,
    ylim=ylim
)

print("Single scenario plots without colorbar completed.")