#!/usr/bin/env -S mamba run -n env_py311 python

import sys
import os

# Add src/ to Python path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from flow_depth_plotting import g_plots_selected_wd_liestal_dinamic_no_cbar

# ─── Parameters ──────────────────────────────────────────────────────────────

dem_file = "/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Liestal_2m/Liestal_2m.dem"
geo_shapefile = "/rs_scratch/users/ge24z347/geo_ezgg_2km_ge.shp"
bounds_file = "/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Liestal_2m/Liestal_2m_4km_bounds.txt"
initial_datetime_str = "2024-06-25T15:00:00"
lead_times_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
color1 = "navajowhite"
color2 = "darkorange"
color3 = "firebrick"

# ─── Read bounding box ───────────────────────────────────────────────────────

with open(bounds_file, "r") as f:
    minx, miny, maxx, maxy = map(float, f.read().strip().split(","))

xlim = (minx, maxx)
ylim = (miny, maxy)

# ─── Loop over ensemble members ──────────────────────────────────────────────

for i in range(1, 12):
    folder_name = f"Liestal_2m_{i}_fv1-gpu"
    wd_folder = f"/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Liestal_2m/{folder_name}/"
    output_folder = f"/storage/homefs/ge24z347/LISFLOOD_FP_outputs/{folder_name}"

    if not os.path.exists(wd_folder):
        print(f"⚠️  Skipping {folder_name} (folder not found)")
        continue

    print(f"📈 Plotting (no colorbar): {folder_name}")

    g_plots_selected_wd_liestal_dinamic_no_cbar(
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

print("✅ All ensemble plots without colorbar completed.")