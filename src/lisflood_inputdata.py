############################ LISFLOOD_FP_ INPUT DATA ###################
######## FOR DIFFERENT VALUES #########################################

import os
import shutil
from DEM_processing import (
    CLIP_DEM_CATCHMENT,
    convert_tif_to_asc,
    rename_file_extension
)

def lisflood_input_dem(buffer_start, buffer_end, buffer_step, chosen_id, 
                       geo_ezgg_shapefile, catchment_csv, dem_file, output_dir):
    """
    Process DEM data with specified buffer distances and chosen_id.
    
    Parameters:
    buffer_start (int): The starting buffer distance in meters.
    buffer_end (int): The ending buffer distance in meters.
    buffer_step (int): The step size for buffer distances.
    chosen_id (int): The ID corresponding to the case study.
    geo_ezgg_shapefile (str): Path to the geo_ezgg shapefile.
    catchment_csv (str): Path to the catchment location CSV file.
    dem_file (str): Path to the input DEM file.
    output_dir (str): Path to the output directory where results will be stored.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract base name from the DEM file and format it correctly
    base_name = "_".join(os.path.splitext(os.path.basename(dem_file))[0].split("_")[:2])  # Ensures "Gordevio_2m"

    # Define buffer distances
    buffer_distances = range(buffer_start, buffer_end + 1, buffer_step)

    # Step 1: Clip DEM to Catchment (Only Once for First Buffer Distance)
    first_buffer = buffer_distances[0]
    output_raster_25 = os.path.join(output_dir, f"{base_name}_catchment_perc999_{first_buffer}.tif")

    print(f" Running CLIP_DEM_CATCHMENT with {first_buffer}m buffer and chosen_id={chosen_id}...")
    CLIP_DEM_CATCHMENT(geo_ezgg_shapefile, catchment_csv, dem_file, output_raster_25, chosen_id=chosen_id, buffer_distance=first_buffer)
    print(f" Clipped DEM saved to {output_raster_25}")

    # Step 2: Convert TIFF to ASCII
    output_asc = os.path.join(output_dir, f"{base_name}_catchment_perc999_{first_buffer}.asc")
    convert_tif_to_asc(output_raster_25, output_asc)

    # Step 3: Rename the ASCII file to `.dem`
    new_file_dem = rename_file_extension(output_asc, ".dem")
    print(f"DEM ASCII file saved as {new_file_dem}")

    # Debugging: Print all files in the directory
    print("🔍 Debugging: Files after first processing:")
    print(os.listdir(output_dir))

    # Loop through buffer distances to rename files
    for buffer_distance in buffer_distances:
        print(f" Processing buffer distance: {buffer_distance}")

        # Rename Clipped DEM File
        new_dem_tif = os.path.join(output_dir, f"{base_name}_catchment_perc999_{buffer_distance}.tif")
        if buffer_distance != first_buffer:
            shutil.copy(output_raster_25, new_dem_tif)
            print(f" Copied {output_raster_25} to {new_dem_tif}")
        else:
            new_dem_tif = output_raster_25

        # Convert TIFF to ASCII
        output_asc_buffered = os.path.join(output_dir, f"{base_name}_catchment_perc999_{buffer_distance}.asc")
        convert_tif_to_asc(new_dem_tif, output_asc_buffered)

        # Rename the ASCII file to `.dem`
        new_file_dem_buffered = rename_file_extension(output_asc_buffered, ".dem")

        print(f"Finished processing buffer {buffer_distance}: {new_file_dem_buffered}")

    # Debugging: Print final files in the directory
    print("🔍 Debugging: Final files in output directory:")
    print(os.listdir(output_dir))

    # Step 4: Delete All `.tif` Files Except the Original Inputs
    print(" Cleaning up: Deleting unnecessary .tif files...")

    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if file.endswith(".tif") and file not in {os.path.basename(dem_file), os.path.basename(output_raster_25)}:
            os.remove(file_path)
            print(f"🗑 Deleted: {file_path}")

    print(" Processing completed!\n")


############################################################################################

############################################################################################

import os
from rasterio.warp import Resampling
from DEM_processing import areal_to_demtarget_resampling, convert_tif_to_asc, rename_file_extension

def lisflood_input_n(dem_file_path, arealstatistik_path, buffer_start, buffer_end, buffer_step):
    """
    Processes Arealstatistik data for multiple precipitation values by:
    - Resampling Arealstatistik to match the DEM file grid.
    - Converting the resampled raster to ASCII.
    - Renaming ASCII files to `.n`.

    Parameters:
    - dem_file_path (str): Path to the DEM raster file (e.g., "path/to/Gordevio_2m_perc999.tif").
    - arealstatistik_path (str): Path to the Arealstatistik raster file (100m resolution).
    - buffer_start (int): Starting precipitation value (e.g., 25).
    - buffer_end (int): Ending precipitation value (e.g., 80).
    - buffer_step (int): Step size (e.g., 10).
    """

    # Extract base name from DEM file (e.g., "Gordevio_2m")
    base_name = "_".join(os.path.splitext(os.path.basename(dem_file_path))[0].split("_")[:2])

    # Define output directory dynamically based on extracted base name
    output_dir = f"/rs_scratch/users/ge24z347/{base_name}/"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Resample Arealstatistik to match DEM grid
    first_precip = buffer_start  # First value in the range
    output_raster_path = os.path.join(output_dir, f"{base_name}_arealstatistik_perc999_{first_precip}.tif")
    
    print(f"Resampling Arealstatistik for {base_name} using first precipitation value: {first_precip} mm...")
    areal_to_demtarget_resampling(dem_file_path, arealstatistik_path, output_raster_path, resampling_method=Resampling.nearest)
    print(f" Resampled Arealstatistik saved to {output_raster_path}")

    # Step 2: Loop through precipitation values to create ASCII & .n files
    for precip in range(buffer_start, buffer_end + 1, buffer_step):
        print(f"Processing {base_name} for precipitation = {precip} mm")

        # Convert the resampled raster to ASCII
        output_asc = os.path.join(output_dir, f"{base_name}_catchment_perc999_{precip}.asc")
        convert_tif_to_asc(output_raster_path, output_asc, desired_nodata_value=-9999)

        # Rename ASCII file to `.n`
        new_file_path = rename_file_extension(output_asc, ".n")
        print(f"Converted {output_asc} → {new_file_path}")

    print(f"All processing completed for {base_name}!\n")
#############################################################################################
####################### Netcdfile ICON croping per ensemble member ##########################
import os
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
from netCDF4 import Dataset
from datetime import date

def crop_icon_to_dem(
    orig_nc: str,
    dem_file: str,
    output_folder: str,
    target_time_str: str = "2024-06-25T00:00:00",
    max_lead_hours: int = 5
):
    """
    Crop ICON precipitation to the exact DEM footprint (no resampling),
    and save one NetCDF per realization with dims (time, y, x),
    time in pure hours (float), and rainfall_depth(time,y,x).
    Filenames will be Liestal_2m_1.nc … Liestal_2m_11.nc
    matching the ICON 'realization' index.
    """
    os.makedirs(output_folder, exist_ok=True)

    # 1) read DEM bounds
    dem = (
        rxr.open_rasterio(dem_file, masked=True)
           .sel(band=1)
           .rio.write_crs("EPSG:2056")
    )
    left, bottom, right, top = dem.rio.bounds()
    print(f"DEM bounds ▶ X {left:.0f}→{right:.0f}, Y {bottom:.0f}→{top:.0f}")

    # 2) open ICON & select your forecast_reference_time + lead_times
    ds = xr.open_dataset(orig_nc)
    ds_sel = (
        ds
        .sel(forecast_reference_time=np.datetime64(target_time_str), method="nearest")
        .sel(lead_time=ds.lead_time <= np.timedelta64(max_lead_hours, "h"))
    )

    # 3) drop old 2D time, convert lead_time → 1D hours
    if "time" in ds_sel.coords:
        ds_sel = ds_sel.drop_vars("time")
    hours = (ds_sel.lead_time / np.timedelta64(1, "h")).astype(np.float32)
    ds_sel = ds_sel.rename_dims({"lead_time": "time"})
    ds_sel = ds_sel.rename_vars({"lead_time": "time"})
    ds_sel = ds_sel.assign_coords(time=hours)

    # 4) slice purely by x/y (no reproject)
    ds_sel = ds_sel.sel(x=slice(left, right), y=slice(bottom, top))
    print("After slice, ICON grid covers exactly your DEM box — no interpolation.")

    # 5) write one file per realization
    for r in ds_sel.realization.values:
        da = (
            ds_sel["precipitation_amount"]
            .sel(realization=r)
            .transpose("time", "y", "x")
            .drop_vars(["forecast_reference_time","realization"], errors="ignore")
        )

        # pull out data as a numpy array
        data = da.values.astype(np.float32)

        # ───── ZERO OUT NaNs ────────────────────────────────────────────────────
        data = np.nan_to_num(data, nan=0.0)
        # ────────────────────────────────────────────────────────────────────────

        t    = da.time.values.astype(np.float32)
        x    = da.x.values.astype(np.float32)
        y    = da.y.values.astype(np.float32)
        nt, ny, nx = data.shape

        out_fn = os.path.join(output_folder, f"Liestal_2m_{int(r+1)}.nc")
        nc = Dataset(out_fn, "w")

        # dimensions
        nc.createDimension("time", nt)
        nc.createDimension("x",    nx)
        nc.createDimension("y",    ny)

        # coords
        tv = nc.createVariable("time","f4",("time",))
        xv = nc.createVariable("x",   "f8",("x",))
        yv = nc.createVariable("y",   "f8",("y",))

        tv.units = "hour"; tv.axis="T"
        xv.units = "m";    xv.axis="X"
        yv.units = "m";    yv.axis="Y"

        tv[:] = t
        xv[:] = x
        yv[:] = y

        # data var
        rv = nc.createVariable(
            "rainfall_depth","f4",
            ("time","y","x"),
            zlib=True, complevel=4, shuffle=True
        )
        rv.units         = "mm"
        rv.standard_name = "precipitation_amount"
        rv[:]            = data

        # globals
        nc.description = f"Cropped ICON rainfall, realization {int(r+1)}"
        nc.history     = f"Created on {date.today().isoformat()}"
        nc.source      = "ICON forecast cropped to DEM box (no resampling)"

        nc.close()
        print(f"✔ Saved: {out_fn}")
        
        
#################################################################################################
######################## COSMO AND ICON BOTH ##################################################

import os
import numpy as np
import xarray as xr
import rioxarray as rxr
from netCDF4 import Dataset
from datetime import date

def _to_hours(arr):
    """Return arr as float32 hours, handling timedelta, datetime, or numeric+units."""
    if np.issubdtype(arr.dtype, np.timedelta64):
        return (arr / np.timedelta64(1, "h")).astype("float32")
    if np.issubdtype(arr.dtype, np.datetime64):
        t0 = arr.values[0]
        return ((arr.values - t0) / np.timedelta64(1, "h")).astype("float32")
    units = (getattr(arr, "attrs", {}) or {}).get("units", "").lower()
    vals = np.asarray(arr.values, dtype="float64")
    if "hour" in units or units == "h":
        return vals.astype("float32")
    if "second" in units or units == "s":
        return (vals / 3600.0).astype("float32")
    if "minute" in units or units in ("min", "m"):
        return (vals / 60.0).astype("float32")
    if np.nanmax(vals) > 1000:              # heuristic: likely seconds
        return (vals / 3600.0).astype("float32")
    return vals.astype("float32")            # assume already hours

def crop_icon_cosmo_to_dem(
    orig_nc: str,
    dem_file: str,
    output_folder: str,
    target_time_str: str = "2024-06-25T00:00:00",
    max_lead_hours: int = 5,
    prefix: str | None = None,
):
    """
    Crop ICON/COSMO precipitation to the DEM bbox (no resampling) and write one
    NetCDF per realization. Dtypes:
      time: float32 hours, x/y: float64 meters, rainfall_depth: float32 mm.
    """

    os.makedirs(output_folder, exist_ok=True)

    # 1) DEM & bounds
    if dem_file.lower().endswith((".dem", ".asc")):
        dem = rxr.open_rasterio(dem_file, masked=True, driver="AAIGrid").sel(band=1)
    else:
        dem = rxr.open_rasterio(dem_file, masked=True).sel(band=1)
    if dem.rio.crs is None:
        dem = dem.rio.write_crs("EPSG:2056")
    left, bottom, right, top = dem.rio.bounds()
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(dem_file))[0]
    print(f"DEM bounds ▶ X {left:.0f}→{right:.0f}, Y {bottom:.0f}→{top:.0f}")

    # 2) Open forecast robustly
    try:
        ds = xr.open_dataset(orig_nc)
    except Exception:
        ds = xr.open_dataset(orig_nc, decode_timedelta=False)

    # 3) Select init time if present
    if "forecast_reference_time" in ds.coords:
        ds = ds.sel(forecast_reference_time=np.datetime64(target_time_str), method="nearest")

    # 4) Normalize time → hours(float32) and clip
    if "lead_time" in ds.coords:
        ds = ds.drop_vars("time", errors="ignore")
        hours_all = _to_hours(ds["lead_time"])
        keep = xr.DataArray(hours_all, coords={"lead_time": ds["lead_time"]}, dims=("lead_time",)) \
               <= float(max_lead_hours)
        ds = ds.sel(lead_time=keep)
        hours_kept = _to_hours(ds["lead_time"]).astype("float32")
        ds = ds.assign_coords(time=("lead_time", hours_kept)).swap_dims({"lead_time": "time"})
    else:
        if "time" not in ds.coords:
            raise ValueError("Dataset has neither 'lead_time' nor 'time'.")
        hours = _to_hours(ds["time"])
        keep = xr.DataArray(hours, coords={"time": ds["time"]}, dims=("time",)) <= float(max_lead_hours)
        ds = ds.sel(time=keep)
        ds = ds.assign_coords(time=("time", _to_hours(ds["time"]).astype("float32")))

    # 5) Spatial crop (no interpolation), handle axis order
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("Dataset must have 'x' and 'y' coordinates.")
    x, y = ds["x"].values, ds["y"].values
    xs = slice(min(left, right), max(left, right)) if x[0] < x[-1] else slice(max(left, right), min(left, right))
    ys = slice(min(bottom, top), max(bottom, top)) if y[0] < y[-1] else slice(max(bottom, top), min(bottom, top))
    ds = ds.sel(x=xs, y=ys)
    print("Spatial crop done (no interpolation).")

    # 6) Variable (assumes 'precipitation_amount'); ensure 'time' dim
    var = ds["precipitation_amount"]
    if "lead_time" in var.dims:
        var = var.rename({"lead_time": "time"})

    # 7) Write one file per realization with enforced dtypes
    reals = ds["realization"].values if "realization" in ds.coords else [None]
    for idx, r in enumerate(reals, start=1):
        sub = var if r is None else var.sel(realization=r)
        da = sub.transpose("time", "y", "x").drop_vars(
            ["forecast_reference_time", "realization"], errors="ignore"
        )

        # ENFORCE DTYPES: time f32, x/y f64, data f32
        data  = np.nan_to_num(da.values, nan=0.0).astype("float32")
        tvals = da.coords["time"].values.astype("float32")
        xvals = da.coords["x"].values.astype("float64")
        yvals = da.coords["y"].values.astype("float64")

        out_fn = os.path.join(output_folder, f"{prefix}_{idx}.nc")
        with Dataset(out_fn, "w") as nc:
            nt, ny, nx = data.shape
            nc.createDimension("time", nt)
            nc.createDimension("x", nx)
            nc.createDimension("y", ny)

            tv = nc.createVariable("time", "f4", ("time",))
            tv.units = "hour"; tv.axis = "T"; tv[:] = tvals

            xv = nc.createVariable("x", "f8", ("x",))
            xv.units = "m"; xv.axis = "X"; xv[:] = xvals

            yv = nc.createVariable("y", "f8", ("y",))
            yv.units = "m"; yv.axis = "Y"; yv[:] = yvals

            rv = nc.createVariable("rainfall_depth", "f4", ("time", "y", "x"),
                                   zlib=True, complevel=4, shuffle=True)
            rv.units = "mm"
            rv.standard_name = "precipitation_amount"
            rv[:] = data

            nc.description = "Cropped ICON rainfall, realization {}".format(idx) if r is not None \
                             else "Cropped ICON rainfall (single realization)"
            nc.history = f"Created on {date.today().isoformat()}"
            nc.source = "ICON forecast cropped to DEM box (no resampling)"

        print(f"✔ Saved: {out_fn}")


##########################################################################################
#####################CROIP COMBIPRECIP WITH DEM ADDED FROM QGIS ########################
##################### NOT WITH DEM MERGED AUTOMATICALLY QUADRATIC FORM TO MATCH NETCDFILE #
#############################################################################################

import numpy as np
import xarray as xr
import rioxarray as rxr
from netCDF4 import Dataset
from datetime import date

def crop_deterministic_Combiprecip(
    orig_nc: str,
    dem_file: str,
    output_nc: str,
    selected_times: list,
    *,
    time_name: str = "REFERENCE_TS",
    var_name: str = "CPC",
    x_name: str = "x",
    y_name: str = "y",
    dem_crs: str = "EPSG:2056",
):
    # --------------------------------------------------
    # 1) DEM bounds
    # --------------------------------------------------
    dem = rxr.open_rasterio(dem_file, masked=True).sel(band=1)
    if not dem.rio.crs:
        dem = dem.rio.write_crs(dem_crs)

    left, bottom, right, top = dem.rio.bounds()
    print(f"DEM bounds ▶ X {left:.0f}→{right:.0f}, Y {bottom:.0f}→{top:.0f}")

    # --------------------------------------------------
    # 2) Open NetCDF and checks
    # --------------------------------------------------
    ds = xr.open_dataset(orig_nc)

    for cname in (time_name, x_name, y_name):
        if cname not in ds.coords:
            raise KeyError(f"Coordinate '{cname}' not found. Available: {list(ds.coords)}")
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found. Available: {list(ds.data_vars)}")

    selected_times = np.array(selected_times, dtype="datetime64[ns]")

    # --------------------------------------------------
    # 3) Compute CPC dx/dy and expand crop by half-cell
    # --------------------------------------------------
    x_vals = ds[x_name].values.astype(float)
    y_vals = ds[y_name].values.astype(float)

    dx = float(np.median(np.diff(np.sort(x_vals))))
    dy = float(np.median(np.diff(np.sort(y_vals))))
    dy_abs = abs(dy)

    print(f"CPC spacing ▶ dx={dx:.1f}  dy={dy:.1f}")

    x_min = left  - dx/2
    x_max = right + dx/2
    y_min = bottom - dy_abs/2
    y_max = top    + dy_abs/2

    x_slice = slice(x_min, x_max) if x_vals[0] < x_vals[-1] else slice(x_max, x_min)
    y_slice = slice(y_max, y_min) if y_vals[0] > y_vals[-1] else slice(y_min, y_max)

    # --------------------------------------------------
    # 4) Subset
    # --------------------------------------------------
    ds_sel = ds.sel({time_name: selected_times}).sel({x_name: x_slice, y_name: y_slice})

    print(
        f"Selected steps: {ds_sel.sizes[time_name]} | "
        f"Crop size: x={ds_sel.sizes[x_name]}, y={ds_sel.sizes[y_name]}"
    )

    # --------------------------------------------------
    # 5) Extract variable
    # --------------------------------------------------
    da = ds_sel[var_name].transpose(time_name, y_name, x_name)

    data = np.nan_to_num(da.values.astype(np.float32), nan=0.0)
    x = da.coords[x_name].values.astype(np.float32)
    y = da.coords[y_name].values.astype(np.float32)

    nt, ny, nx = data.shape

    # --------------------------------------------------
    # 6) Write NetCDF
    # --------------------------------------------------
    nc = Dataset(output_nc, "w")

    nc.createDimension("time", nt)
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)

    # ✅ TIME (FIXED)
    tv = nc.createVariable("time", "f8", ("time",))
    #tv.long_name = "time"
    tv.units = "hour"
    tv.axis = "T"
    tv[:] = np.arange(nt, dtype=np.float64)

    # SPACE
    xv = nc.createVariable("x", "f4", ("x",))
    yv = nc.createVariable("y", "f4", ("y",))
    xv.units = "m"; xv.axis = "X"
    yv.units = "m"; yv.axis = "Y"
    xv[:] = x
    yv[:] = y

    # ✅ RAINFALL (clean, only units)
    rv = nc.createVariable(
        "rainfall_depth", "f4", ("time", "y", "x"),
        zlib=True, complevel=4, shuffle=True
    )
    rv.units = "mm"
    rv[:] = data

    nc.description = "Cropped deterministic CPC rainfall"
    nc.history = f"Created on {date.today().isoformat()}"
    nc.source = "CPC deterministic forecast cropped to DEM"

    nc.close()
    ds.close()

    print(f"✔ Saved: {output_nc}")


#######################################################################################
###########################CROP COMBIPRECIP BIGGER AS THE TIF I GAVE IN THE INPUT#####
########################################################################################

import numpy as np
import xarray as xr
import rioxarray as rxr
from netCDF4 import Dataset
from datetime import date

def crop_deterministic_Combiprecip_inf(
    orig_nc: str,
    dem_file: str,
    output_nc: str,
    selected_times: list,
    *,
    time_name: str = "REFERENCE_TS",
    var_name: str = "CPC",
    x_name: str = "x",
    y_name: str = "y",
    dem_crs: str = "EPSG:2056",
    rain_factor: float = 1.0,  # NEW: multiplicative factor applied to rainfall_depth
):
    # --------------------------------------------------
    # 1) DEM bounds
    # --------------------------------------------------
    dem = rxr.open_rasterio(dem_file, masked=True).sel(band=1)
    if not dem.rio.crs:
        dem = dem.rio.write_crs(dem_crs)

    left, bottom, right, top = dem.rio.bounds()
    print(f"DEM bounds ▶ X {left:.0f}→{right:.0f}, Y {bottom:.0f}→{top:.0f}")

    # --------------------------------------------------
    # 2) Open NetCDF and checks
    # --------------------------------------------------
    ds = xr.open_dataset(orig_nc)

    for cname in (time_name, x_name, y_name):
        if cname not in ds.coords:
            raise KeyError(f"Coordinate '{cname}' not found. Available: {list(ds.coords)}")
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found. Available: {list(ds.data_vars)}")

    selected_times = np.array(selected_times, dtype="datetime64[ns]")

    # --------------------------------------------------
    # 3) Compute CPC dx/dy and expand crop by half-cell
    # --------------------------------------------------
    x_vals = ds[x_name].values.astype(float)
    y_vals = ds[y_name].values.astype(float)

    if len(x_vals) < 2 or len(y_vals) < 2:
        raise ValueError("x/y coordinates too short to infer dx/dy")

    dx = float(np.median(np.diff(np.sort(x_vals))))
    dy = float(np.median(np.diff(np.sort(y_vals))))
    dy_abs = abs(dy)

    print(f"CPC spacing ▶ dx={dx:.1f}  dy={dy:.1f}")

    # Expand DEM bbox by half a CPC cell so outer CPC cells are included
    x_min = left  - dx/2
    x_max = right + dx/2
    y_min = bottom - dy_abs/2
    y_max = top    + dy_abs/2

    # Respect axis direction for slice
    x_slice = slice(x_min, x_max) if x_vals[0] < x_vals[-1] else slice(x_max, x_min)
    y_slice = slice(y_max, y_min) if y_vals[0] > y_vals[-1] else slice(y_min, y_max)

    # --------------------------------------------------
    # 4) Subset time + expanded space
    # --------------------------------------------------
    ds_sel = ds.sel({time_name: selected_times}).sel({x_name: x_slice, y_name: y_slice})

    if ds_sel.sizes.get(time_name, 0) == 0:
        raise ValueError("No matching time steps after selection.")
    if ds_sel.sizes.get(x_name, 0) == 0 or ds_sel.sizes.get(y_name, 0) == 0:
        raise ValueError("Spatial crop returned empty x or y dimension.")

    print(
        f"Selected steps: {ds_sel.sizes[time_name]} | "
        f"Crop size: x={ds_sel.sizes[x_name]}, y={ds_sel.sizes[y_name]}"
    )

    # --- Print resulting bounds (center coords)
    x_out = ds_sel[x_name].values.astype(float)
    y_out = ds_sel[y_name].values.astype(float)

    print("NC bounds (centers):",
          (float(x_out.min()), float(y_out.min()), float(x_out.max()), float(y_out.max())))

    # If you want “cell-edge bbox” (more meaningful for raster extents)
    out_left   = float(x_out.min() - dx/2)
    out_right  = float(x_out.max() + dx/2)
    out_bottom = float(y_out.min() - dy_abs/2)
    out_top    = float(y_out.max() + dy_abs/2)
    print("NC bbox (edges)  ≈", (out_left, out_bottom, out_right, out_top))

    # --------------------------------------------------
    # 5) Extract variable and reorder dims → (time, y, x)
    # --------------------------------------------------
    da = ds_sel[var_name].transpose(time_name, y_name, x_name)

    data = np.nan_to_num(da.values.astype(np.float32), nan=0.0)

    # NEW: apply multiplicative factor to all time steps
    rain_factor = float(rain_factor)
    data *= np.float32(rain_factor)
    print(f"Applied rain_factor={rain_factor}")

    x = da.coords[x_name].values.astype(np.float32)
    y = da.coords[y_name].values.astype(np.float32)

    nt, ny, nx = data.shape

    # --------------------------------------------------
    # 6) Write NetCDF
    # --------------------------------------------------
    nc = Dataset(output_nc, "w")

    nc.createDimension("time", nt)
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)

    # time float64: 0.0..N-1
    tv = nc.createVariable("time", "f8", ("time",))
    tv.long_name = "time_step"
    tv.units = "1"
    tv.axis = "T"
    tv[:] = np.arange(nt, dtype=np.float64)

    xv = nc.createVariable("x", "f4", ("x",))
    yv = nc.createVariable("y", "f4", ("y",))
    xv.units = "m"; xv.axis = "X"
    yv.units = "m"; yv.axis = "Y"
    xv[:] = x
    yv[:] = y

    rv = nc.createVariable(
        "rainfall_depth", "f4", ("time", "y", "x"),
        zlib=True, complevel=4, shuffle=True
    )
    rv.units = "mm"
    rv.standard_name = "precipitation_amount"
    rv[:] = data

    nc.description = "Cropped deterministic CPC rainfall (half-cell expanded bbox)"
    nc.history = f"Created on {date.today().isoformat()}; rain_factor={rain_factor}"
    nc.source = "CPC deterministic forecast cropped to DEM footprint"

    nc.close()
    ds.close()

    print(f"✔ Saved: {output_nc}")
    print("✔ time written as float64: 0.0..N-1")


##########################################################################################
#####################CROP COMBIPRECIP EXACT AS THE TIF OR DEM  ########################
######################
#############################################################################################
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from datetime import date
import os, math

def crop_deterministic_Combiprecip_bounds(
    orig_nc: str,
    dem_file: str,
    output_nc: str,
    selected_times: list,
    *,
    time_name: str = "REFERENCE_TS",
    var_name: str = "CPC",
    x_name: str = "x",
    y_name: str = "y",
    snap_res: int = 1000,   # resolution of CPC grid
):
    """
    Crop a Combiprecip NetCDF to DEM footprint using *_bounds.txt,
    snapping DEM bounds to CPC grid resolution.
    """

    # 1) Load bounds from *_bounds.txt
    bounds_txt = dem_file.replace(".tif", "_bounds.txt").replace(".dem", "_bounds.txt")
    if not os.path.exists(bounds_txt):
        raise FileNotFoundError(f"Bounds file not found: {bounds_txt}")

    with open(bounds_txt, "r", encoding="utf-8-sig") as f:
        left, bottom, right, top = map(float, f.read().strip().split(","))

    # Snap bounds to multiples of snap_res (CPC resolution)
    left   = math.floor(left   / snap_res) * snap_res
    bottom = math.floor(bottom / snap_res) * snap_res
    right  = math.ceil(right  / snap_res) * snap_res
    top    = math.ceil(top    / snap_res) * snap_res

    print(f"Snapped bounds: X {left}→{right}, Y {bottom}→{top}")

    # 2) Open NetCDF
    ds = xr.open_dataset(orig_nc)

    # sanity checks
    for cname in (time_name, x_name, y_name):
        if cname not in ds.coords:
            raise KeyError(f"Coordinate '{cname}' not found in {list(ds.coords)}")
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in {list(ds.data_vars)}")

    # Normalize time array
    selected_times = np.array(selected_times, dtype="datetime64[ns]")

    # Slicing that respects axis direction
    x_vals = ds[x_name].values
    y_vals = ds[y_name].values
    x_slice = slice(left, right) if x_vals[0] < x_vals[-1] else slice(right, left)
    y_slice = slice(top, bottom) if y_vals[0] > y_vals[-1] else slice(bottom, top)

    # 3) Subset in time & space
    ds_sel = (
        ds.sel({time_name: selected_times})
          .sel({x_name: x_slice, y_name: y_slice})
    )
    if ds_sel.sizes.get(time_name, 0) == 0:
        raise ValueError("No matching time steps after selection.")

    print(f"Selected {ds_sel.sizes[time_name]} steps; spatial crop done.")

    # 4) Extract variable and reorder dims → (time, y, x)
    da = ds_sel[var_name]
    da = da.transpose(time_name, y_name, x_name)
    data = np.nan_to_num(da.values.astype(np.float32), nan=0.0)
    x = da.coords[x_name].values.astype(np.float32)
    y = da.coords[y_name].values.astype(np.float32)
    nt, ny, nx = data.shape

    # 5) Write NetCDF
    nc = Dataset(output_nc, "w")

    nc.createDimension("time", nt)
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)

    tv = nc.createVariable("time", "f8", ("time",))
    xv = nc.createVariable("x", "f4", ("x",))
    yv = nc.createVariable("y", "f4", ("y",))

    tv.units = "seconds since 1970-01-01 00:00:00"
    tv.axis = "T"
    xv.units = "m"; xv.axis = "X"
    yv.units = "m"; yv.axis = "Y"

    tv[:] = np.arange(nt, dtype=np.float32)
    xv[:] = x
    yv[:] = y

    rv = nc.createVariable(
        "rainfall_depth", "f4", ("time", "y", "x"),
        zlib=True, complevel=4, shuffle=True
    )
    rv.units = "mm"
    rv.standard_name = "precipitation_amount"
    rv[:] = data

    nc.description = "Cropped deterministic CPC rainfall"
    nc.history     = f"Created on {date.today().isoformat()}"
    nc.source      = "CPC deterministic forecast cropped to DEM bounds (snapped)"
    nc.close()

    ds.close()
    print(f"✔ Saved: {output_nc}")


### EXAMPLE TO COPY AND RENAME THE FILE WITH DIFFERENT NAMES ##################################
import os
import shutil

def copy_and_rename_n_file(base_file, output_dir, start, end, step):
    """
    Copies a .n file and renames it for multiple precipitation values.
    
    Parameters:
        base_file (str): Path to the original .n file.
        output_dir (str): Directory where the renamed files will be saved.
        start (int): Starting precipitation value.
        end (int): Ending precipitation value.
        step (int): Step size for precipitation values.
    """
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base file not found: {base_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(base_file))[0]  # Extract base name without extension
    
    for value in range(start, end + 1, step):
        new_file = os.path.join(output_dir, f"{base_name}_{value}.n")
        shutil.copy(base_file, new_file)
        print(f" Copied and renamed: {new_file}")

###################################################################################################

def copy_and_rename_dem_file(base_file, output_dir, start, end, step):
    """
    Copies a .n file and renames it for multiple precipitation values.
    
    Parameters:
        base_file (str): Path to the original .n file.
        output_dir (str): Directory where the renamed files will be saved.
        start (int): Starting precipitation value.
        end (int): Ending precipitation value.
        step (int): Step size for precipitation values.
    """
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base file not found: {base_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(base_file))[0]  # Extract base name without extension
    
    for value in range(start, end + 1, step):
        new_file = os.path.join(output_dir, f"{base_name}_{value}.dem")
        shutil.copy(base_file, new_file)
        print(f"Copied and renamed: {new_file}")


#####################################################################################################

import os
from DEM_processing import precipitation_netcdf

def create_precipitation_netcdfs(dem_file_path, buffer_start, buffer_end, buffer_step):
    """
    Generate NetCDF files for all buffer distances for a given case study.

    Parameters:
    - dem_file_path (str): Path to the DEM raster file (e.g., "/path/to/Gordevio_2m_perc999.tif").
    - buffer_start (int): The starting precipitation value in mm.
    - buffer_end (int): The ending precipitation value in mm.
    - buffer_step (int): The step size for precipitation values.
    """

    # Extract base name from DEM file (e.g., "Gordevio_2m")
    base_name = "_".join(os.path.splitext(os.path.basename(dem_file_path))[0].split("_")[:2])

    # Define output directory dynamically based on extracted base name
    output_dir = f"/rs_scratch/users/ge24z347/{base_name}/"
    os.makedirs(output_dir, exist_ok=True)

    # Define frequency per hour distribution (same for all cases)
    frequency_per_hour = [0, 9.30, 17.81, 18.70, 14.37, 11.13, 7.43, 5.28, 4.47, 3.41, 3.01, 2.68, 2.41]

    # Loop through the precipitation values (e.g., 25mm to 80mm, step 5mm)
    for total_precipitation in range(buffer_start, buffer_end + 1, buffer_step):
        print(f" Creating NetCDF for {base_name} with total precipitation = {total_precipitation} mm")

        # Define input DEM file
        dem_file = os.path.join(output_dir, f"{base_name}_catchment_perc999_{total_precipitation}.dem")

        # Define output NetCDF file
        nc_path = os.path.join(output_dir, f"{base_name}_catchment_perc999_{total_precipitation}.nc")

        # Run precipitation NetCDF generation
        precipitation_netcdf(dem_file, total_precipitation, nc_path, frequency_per_hour)

        print(f"Created NetCDF: {nc_path}")

    print(f" All NetCDF files created for {base_name}_catchment_perc999_{total_precipitation}!\n")


##################################################################################################

import os
from DEM_processing import precipitation_netcdf

def create_precipitation_netcdfs_quadratic(dem_file_path, buffer_start, buffer_end, buffer_step):
    """
    Generate NetCDF files for all buffer distances for a given case study.

    Parameters:
    - dem_file_path (str): Path to the DEM raster file (e.g., "/path/to/Gordevio_2m_perc999.tif").
    - buffer_start (int): The starting precipitation value in mm.
    - buffer_end (int): The ending precipitation value in mm.
    - buffer_step (int): The step size for precipitation values.
    """

    # Extract base name from DEM file (e.g., "Gordevio_2m")
    base_name = "_".join(os.path.splitext(os.path.basename(dem_file_path))[0].split("_")[:2])

    # Define output directory dynamically based on extracted base name
    output_dir = f"/rs_scratch/users/ge24z347/{base_name}/"
    os.makedirs(output_dir, exist_ok=True)

    # Define frequency per hour distribution (same for all cases)
    frequency_per_hour = [0, 9.30, 17.81, 18.70, 14.37, 11.13, 7.43, 5.28, 4.47, 3.41, 3.01, 2.68, 2.41]

    # Loop through the precipitation values (e.g., 25mm to 80mm, step 5mm)
    for total_precipitation in range(buffer_start, buffer_end + 1, buffer_step):
        print(f"Creating NetCDF for {base_name} with total precipitation = {total_precipitation} mm")

        # Define input DEM file
        dem_file = os.path.join(output_dir, f"{base_name}_{total_precipitation}.dem")

        # Define output NetCDF file
        nc_path = os.path.join(output_dir, f"{base_name}_{total_precipitation}.nc")

        # Run precipitation NetCDF generation
        precipitation_netcdf(dem_file, total_precipitation, nc_path, frequency_per_hour)

        print(f"Created NetCDF: {nc_path}")

    print(f"All NetCDF files created for {base_name}_{total_precipitation}!\n")

#######################################################################################################

import os
from DEM_processing import  create_stage_file

def various_stage_files(dem_file_path, selected_id, buffer_start, buffer_end, buffer_step, num_points=1):
    """
    Generate .stage files for all buffer distances for a given case study.

    Parameters:
    - dem_file_path (str): Path to any DEM file (used only to extract folder and naming).
    - selected_id (int): The ID from the CSV file corresponding to the case study.
    - buffer_start (int): The starting buffer distance in meters.
    - buffer_end (int): The ending buffer distance in meters.
    - buffer_step (int): The step size for buffer distances.
    - num_points (int): Number of points to include in the .stage file (default: 1).
    """

    # Get folder name to build base name (e.g., "Gordevio_2m")
    folder_name = os.path.basename(os.path.dirname(dem_file_path))

    #  Define base name for files (as requested)
    base_name = f"{folder_name}_catchment_perc999"

    # Set output directory to the same folder as the DEM
    output_dir = os.path.dirname(dem_file_path)
    os.makedirs(output_dir, exist_ok=True)

    #  Path to your CSV with coordinates
    catchment_location_csv = "/rs_scratch/users/ge24z347/catchment_location.csv"

    print(f"Output directory: {output_dir}")
    print(f"File base name: {base_name}")

    # Loop through each buffer size and create the .stage files
    for buffer_distance in range(buffer_start, buffer_end + 1, buffer_step):
        stage_file = os.path.join(output_dir, f"{base_name}_{buffer_distance}.stage")
        print(f" Creating: {os.path.basename(stage_file)}")

        # Call your stage file creation function
        create_stage_file(catchment_location_csv, selected_id, stage_file, num_points)

        print(f" Done: {stage_file}")

    print(f" All .stage files created in: {output_dir}")

############################################################################################################################

import os
from DEM_processing import  create_stage_file

def various_stage_files_quadratic(dem_file_path, selected_id, buffer_start, buffer_end, buffer_step, num_points=1):
    """
    Generate .stage files for all buffer distances for a given case study.

    Parameters:
    - dem_file_path (str): Path to any DEM file (used only to extract folder and naming).
    - selected_id (int): The ID from the CSV file corresponding to the case study.
    - buffer_start (int): The starting buffer distance in meters.
    - buffer_end (int): The ending buffer distance in meters.
    - buffer_step (int): The step size for buffer distances.
    - num_points (int): Number of points to include in the .stage file (default: 1).
    """

    # Get folder name to build base name (e.g., "Gordevio_2m")
    folder_name = os.path.basename(os.path.dirname(dem_file_path))

    # Define base name for files (as requested)
    base_name = f"{folder_name}"

    # Set output directory to the same folder as the DEM
    output_dir = os.path.dirname(dem_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Path to your CSV with coordinates
    catchment_location_csv = "/rs_scratch/users/ge24z347/catchment_location.csv"

    print(f" Output directory: {output_dir}")
    print(f" File base name: {base_name}")

    #  Loop through each buffer size and create the .stage files
    for buffer_distance in range(buffer_start, buffer_end + 1, buffer_step):
        stage_file = os.path.join(output_dir, f"{base_name}_{buffer_distance}.stage")
        print(f" Creating: {os.path.basename(stage_file)}")

        # Call your stage file creation function
        create_stage_file(catchment_location_csv, selected_id, stage_file, num_points)

        print(f" Done: {stage_file}")

    print(f" All .stage files created in: {output_dir}")


###############################################################################################################################

import os
from DEM_processing import create_par_file_fv1

def various_par_files_fv1(dem_file_path, buffer_start, buffer_end, buffer_step):
    """
    Generate .par files for different precipitation values for a given case study.
    
    Parameters:
    - dem_file_path (str): Path to a DEM file (used to extract folder and base name).
    - buffer_start (int): Starting precipitation value in mm.
    - buffer_end (int): Ending precipitation value in mm.
    - buffer_step (int): Step size for precipitation values.
    """

    # Get folder name (e.g., "Gordevio_2m")
    folder_name = os.path.basename(os.path.dirname(dem_file_path))

    # Create base name (e.g., "Gordevio_2m")
    base_prefix = f"{folder_name}_"

    # Output directory (same as DEM folder)
    output_dir = os.path.dirname(dem_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Loop through precipitation values and create .par files
    for total_precipitation in range(buffer_start, buffer_end + 1, buffer_step):
        base_name = f"{base_prefix}_{total_precipitation}"  # e.g. Gordevio_2m_45
        output_par_file = os.path.join(output_dir, f"{base_name}.par")

        print(f" Creating .par file: {os.path.basename(output_par_file)}")

        # Call your function to create the .par file
        create_par_file_fv1(base_name, str(total_precipitation), output_par_file)

        print(f" Created: {output_par_file}")

    print(f" All .par files created in: {output_dir}")

###############################################################################################################

import os
from DEM_processing import create_par_file_acc

def various_par_files_acc(dem_file_path, buffer_start, buffer_end, buffer_step):
    """
    Generate .par files for different precipitation values for a given case study.
    
    Parameters:
    - dem_file_path (str): Path to a DEM file (used to extract folder and base name).
    - buffer_start (int): Starting precipitation value in mm.
    - buffer_end (int): Ending precipitation value in mm.
    - buffer_step (int): Step size for precipitation values.
    """

    # Get folder name (e.g., "Gordevio_2m")
    folder_name = os.path.basename(os.path.dirname(dem_file_path))

    # Create base name (e.g., "Gordevio_2m")
    base_prefix = f"{folder_name}"

    # Output directory (same as DEM folder)
    output_dir = os.path.dirname(dem_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Loop through precipitation values and create .par files
    for total_precipitation in range(buffer_start, buffer_end + 1, buffer_step):
        base_name = f"{base_prefix}_{total_precipitation}"  # e.g. Gordevio_2m
        output_par_file = os.path.join(output_dir, f"{base_name}.par")

        print(f" Creating .par file: {os.path.basename(output_par_file)}")

        # Call your function to create the .par file
        create_par_file_acc(base_name, str(total_precipitation), output_par_file)

        print(f"Created: {output_par_file}")


####################################################################################################

import os
from DEM_processing import create_par_file

def various_par_files(dem_file_path, buffer_start, buffer_end, buffer_step):
    """
    Generate .par files for different precipitation values for a given case study.
    
    Parameters:
    - dem_file_path (str): Path to a DEM file (used to extract folder and base name).
    - buffer_start (int): Starting precipitation value in mm.
    - buffer_end (int): Ending precipitation value in mm.
    - buffer_step (int): Step size for precipitation values.
    """

    # Get folder name (e.g., "Gordevio_2m")
    folder_name = os.path.basename(os.path.dirname(dem_file_path))

    # Create base name (e.g., "Gordevio_2m")
    base_prefix = f"{folder_name}"

    # Output directory (same as DEM folder)
    output_dir = os.path.dirname(dem_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Loop through precipitation values and create .par files
    for total_precipitation in range(buffer_start, buffer_end + 1, buffer_step):
        base_name = f"{base_prefix}_{total_precipitation}"  # e.g. Gordevio_2m
        output_par_file = os.path.join(output_dir, f"{base_name}.par")

        print(f"Creating .par file: {os.path.basename(output_par_file)}")

        # Call your function to create the .par file
        create_par_file(base_name, str(total_precipitation), output_par_file)

        print(f"Created: {output_par_file}")

    print(f" All .par files created in: {output_dir}")

#############################################################################################

import os
from DEM_processing import create_par_file_Liestal_5hour

def various_par_files_Liestal_5hour(dem_file_path, buffer_start, buffer_end, buffer_step):
    """
    Generate .par files for different precipitation values for a given case study.
    
    Parameters:
    - dem_file_path (str): Path to a DEM file (used to extract folder and base name).
    - buffer_start (int): Starting precipitation value in mm.
    - buffer_end (int): Ending precipitation value in mm.
    - buffer_step (int): Step size for precipitation values.
    """

    # Get folder name (e.g., "Gordevio_2m")
    folder_name = os.path.basename(os.path.dirname(dem_file_path))

    # Create base name (e.g., "Gordevio_2m")
    base_prefix = f"{folder_name}"

    # Output directory (same as DEM folder)
    output_dir = os.path.dirname(dem_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Loop through precipitation values and create .par files
    for total_precipitation in range(buffer_start, buffer_end + 1, buffer_step):
        base_name = f"{base_prefix}_{total_precipitation}"  # e.g. Gordevio_2m
        output_par_file = os.path.join(output_dir, f"{base_name}.par")

        print(f"Creating .par file: {os.path.basename(output_par_file)}")

        # Call your function to create the .par file
        create_par_file_Liestal_5hour(base_name, str(total_precipitation), output_par_file)

        print(f"Created: {output_par_file}")

    print(f" All .par files created in: {output_dir}")


####################################################################################

import os
from DEM_processing import create_par_file_Liestal_33hour

def various_par_files_Liestal_33hour(dem_file_path, buffer_start, buffer_end, buffer_step):
    """
    Generate .par files for different precipitation values for a given case study.
    
    Parameters:
    - dem_file_path (str): Path to a DEM file (used to extract folder and base name).
    - buffer_start (int): Starting precipitation value in mm.
    - buffer_end (int): Ending precipitation value in mm.
    - buffer_step (int): Step size for precipitation values.
    """

    # Get folder name (e.g., "Gordevio_2m")
    folder_name = os.path.basename(os.path.dirname(dem_file_path))

    # Create base name (e.g., "Gordevio_2m")
    base_prefix = f"{folder_name}"

    # Output directory (same as DEM folder)
    output_dir = os.path.dirname(dem_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Loop through precipitation values and create .par files
    for total_precipitation in range(buffer_start, buffer_end + 1, buffer_step):
        base_name = f"{base_prefix}_{total_precipitation}"  # e.g. Gordevio_2m
        output_par_file = os.path.join(output_dir, f"{base_name}.par")

        print(f"Creating .par file: {os.path.basename(output_par_file)}")

        # Call your function to create the .par file
        create_par_file_Liestal_33hour(base_name, str(total_precipitation), output_par_file)

        print(f"Created: {output_par_file}")

    print(f" All .par files created in: {output_dir}")

##########################################################################################################

import os
from DEM_processing import create_par_file_Liestal_10hour

def various_par_files_Liestal_10hour(dem_file_path, buffer_start, buffer_end, buffer_step):
    """
    Generate .par files for different precipitation values for a given case study.
    
    Parameters:
    - dem_file_path (str): Path to a DEM file (used to extract folder and base name).
    - buffer_start (int): Starting precipitation value in mm.
    - buffer_end (int): Ending precipitation value in mm.
    - buffer_step (int): Step size for precipitation values.
    """

    # Get folder name (e.g., "Gordevio_2m")
    folder_name = os.path.basename(os.path.dirname(dem_file_path))

    # Create base name (e.g., "Gordevio_2m")
    base_prefix = f"{folder_name}"

    # Output directory (same as DEM folder)
    output_dir = os.path.dirname(dem_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Loop through precipitation values and create .par files
    for total_precipitation in range(buffer_start, buffer_end + 1, buffer_step):
        base_name = f"{base_prefix}_{total_precipitation}"  # e.g. Gordevio_2m
        output_par_file = os.path.join(output_dir, f"{base_name}.par")

        print(f"Creating .par file: {os.path.basename(output_par_file)}")

        # Call your function to create the .par file
        create_par_file_Liestal_10hour(base_name, str(total_precipitation), output_par_file)

        print(f"Created: {output_par_file}")

    print(f" All .par files created in: {output_dir}")

#########################################################################################

########### bci coundary conditions for Combiprecip one determinitsic approach ####################
##################################################################################################

##############################################################################################################################
##############################################################################################################################
######################### add the QFIX as input and not anymore calculated as before ######################################

def write_bci_qflex(
    output_path,
    Q_m3s=None,                        # total inflow discharge (m^3/s); required if you provide any inflows
    cell_size=None,                    # grid cell size (m); required if you provide point inflows
    point_inflows=None,                # list of (x, y) tuples for P inflows, e.g. [(2647230.071, 1177404.771), ...]
    line_inflows=None,                 # list of dicts: [{"side":"E","start":1177400.0,"end":1177520.0}, ...]
    outflow_side="E",                  # FREE outflow is always written
    outflow_start=None,
    outflow_end=None,
    outflow_slope=None                 # optional numeric slope after FREE
):
    """
    Write a .bci that can contain:
      - zero or more point inflows (P x y QFIX <q_per_point>)
      - zero or more line inflows (W/E/N/S start end QFIX <q_per_width>)
      - exactly one FREE outflow (mandatory)

    Rules:
      * QFIX value is flux per unit width (m^2/s).
      * Points: width = cell_size. If you have N points, each gets Q_total / (N * cell_size).
      * Lines: width = segment length. If you have segments with lengths L_i, each gets Q_total / sum(L_i) (same q for all segments).
      * Mixed: split by effective width: W_eff = N_points*cell_size + sum(L_i). Then:
            q_point = Q_total / W_eff / cell_size
            q_line  = Q_total / W_eff    (per unit width)
      * If no inflows are provided, only the FREE outflow line is written.
    """
    lines = []

    # --- validate outflow
    if outflow_start is None or outflow_end is None:
        raise ValueError("outflow_start and outflow_end are required.")

    # --- normalize inputs
    point_inflows = list(point_inflows or [])
    line_inflows  = list(line_inflows or [])

    have_inflows = bool(point_inflows or line_inflows)

    if have_inflows and Q_m3s is None:
        raise ValueError("Q_m3s must be provided when point_inflows or line_inflows are used.")

    if point_inflows and (cell_size is None or cell_size <= 0):
        raise ValueError("cell_size (>0) is required when using point_inflows.")

    # --- compute effective widths
    n_points = len(point_inflows)
    sum_L = 0.0
    if line_inflows:
        for seg in line_inflows:
            a = float(seg["start"])
            b = float(seg["end"])
            L = abs(b - a)
            if L <= 0:
                raise ValueError(f"Line inflow segment length must be > 0, got {L} for {seg}.")
            sum_L += L

    # --- compute per-unit-width q values when needed
    if have_inflows:
        if n_points and sum_L == 0:
            # points only
            q_per_point = Q_m3s / (n_points * float(cell_size))   # m^2/s
            q_per_line = None
        elif sum_L and n_points == 0:
            # lines only
            q_per_line = Q_m3s / sum_L                            # m^2/s
            q_per_point = None
        else:
            # mixed points and lines
            W_eff = n_points * float(cell_size) + sum_L
            if W_eff <= 0:
                raise ValueError("Effective width computed as zero; check inputs.")
            q_per_point = Q_m3s / W_eff / float(cell_size)        # m^2/s
            q_per_line  = Q_m3s / W_eff                           # m^2/s

    # --- emit point inflows
    for (x, y) in point_inflows:
        lines.append(f"P {float(x):.3f} {float(y):.3f} QFIX {q_per_point:.3f}")

    # --- emit line inflows
    for seg in line_inflows:
        side = seg["side"].upper()
        a = float(seg["start"]); b = float(seg["end"])
        if sum_L and n_points == 0:
            qfix = q_per_line
        elif sum_L and n_points:
            qfix = q_per_line
        else:
            # safety: points-only path should not be here
            raise RuntimeError("Internal: line inflow without computed q_per_line.")
        lines.append(f"{side} {a:.3f} {b:.3f} QFIX {qfix:.3f}")

    # --- outflow FREE
    if outflow_slope is None:
        lines.append(f"{outflow_side} {float(outflow_start):.3f} {float(outflow_end):.3f} FREE")
    else:
        lines.append(f"{outflow_side} {float(outflow_start):.3f} {float(outflow_end):.3f} FREE {float(outflow_slope):.6f}")

    # --- write
    with open(output_path, "w") as f:
        for L in lines:
            f.write(L + "\n")

    print(f" .bci written → {output_path}")

####################################################################################

def write_bci_qvar(
    output_path,
    inflows=None,             # dict: {"QPF": [(x,y),...], "QPE": [(x,y),...]}
    outflow_side="E",         # FREE outflow (side) is always written
    outflow_start=None,
    outflow_end=None,
    outflow_slope=None,
    outflow_points_free=None  # NEW: list of (x,y) for point FREE outflows
):
    """
    Write a .bci file with QVAR inflows, one FREE side outflow, and optional FREE point outflows.

    inflows: dict mapping inflow_name -> list of (x,y) points
             e.g. {"QPF":[(2704418.096,1256357.968), (2704420.096,1256357.968)],
                   "QPE":[(...), (...)]}
    outflow_points_free: list of (x,y) to emit as 'P x y FREE'
    """
    if outflow_start is None or outflow_end is None:
        raise ValueError("outflow_start and outflow_end are required.")

    lines = []

    # --- QVAR point inflows
    if inflows:
        for inflow_name, points in inflows.items():
            for (x, y) in points:
                lines.append(f"P {float(x):.3f} {float(y):.3f} QVAR {inflow_name}")

    # --- FREE side outflow
    if outflow_slope is None:
        lines.append(f"{outflow_side} {float(outflow_start):.3f} {float(outflow_end):.3f} FREE")
    else:
        lines.append(
            f"{outflow_side} {float(outflow_start):.3f} {float(outflow_end):.3f} FREE {float(outflow_slope):.6f}"
        )

    # --- FREE point outflows
    if outflow_points_free:
        for (x, y) in outflow_points_free:
            lines.append(f"P {float(x):.3f} {float(y):.3f} FREE")

    # --- write
    with open(output_path, "w") as f:
        f.writelines(L + "\n" for L in lines)

    print(f"✔ .bci written → {output_path}")
    
################################################################################
#################################################################################

#################################################################################
########### bdy file ##########################################################
from pathlib import Path

def write_bdy_qvar(output_path, inflows, times):
    """
    Write a .bdy file for LISFLOOD with QVAR inflows.

    Parameters
    ----------
    output_path : str
        Path to output .bdy file
    inflows : dict
        Dictionary of inflows, e.g.
        {
          "QPF": {"width": 10, "discharges": [0,0,0,0.5,9,4.5,2.5,0]},
          "QPE": {"width": 10, "discharges": [0,...]},
        }
    times : list[int]
        Time steps in seconds (must match length of discharges for each inflow)
    """
    lines = []
    for name, cfg in inflows.items():
        width = cfg["width"]
        discharges = cfg["discharges"]

        if len(discharges) != len(times):
            raise ValueError(f"Inflow {name} discharges length {len(discharges)} "
                             f"!= times length {len(times)}")

        # header
        lines.append("#inflow")
        lines.append(f"## Values incremented = m³/s ÷ {width} = QVAR (m²/s)")
        lines.append(name)
        lines.append(f"{len(times)} seconds")

        # rows
        for q, t in zip(discharges, times):
            qvar = q / width
            lines.append(f"{qvar:.3f}\t{t}")

        lines.append("")  # blank line after each inflow

    Path(output_path).write_text("\n".join(lines))
    print(f"✔ .bdy written → {output_path}")
    
    
    
################################################################################################
############################## create netcdfiles over the dem 60 min and 75 min ###############
################################ uniform ######################################################

import os
from DEM_processing import precipitation_netcdf

def create_precip_automatic_camp(base_name, dem_file_path, buffer_start, buffer_end, buffer_step):
    """
    Generate NetCDF files for different precipitation intensities based on a single DEM file.

    Parameters:
    - base_name (str): Base name for output files (e.g., 'Morges_2m_v1')
    - dem_file_path (str): Path to the DEM file
    - buffer_start (int): Minimum precipitation (e.g., 5)
    - buffer_end (int): Maximum precipitation (e.g., 75)
    - buffer_step (int): Step size (e.g., 10 → 10, 20, 30, ...)
    """
    output_dir = os.path.dirname(dem_file_path)
    os.makedirs(output_dir, exist_ok=True)

    frequency_per_hour = [0, 9.30, 17.81, 18.70, 14.37, 11.13, 7.43, 5.28, 4.47, 3.41, 3.01, 2.68, 2.41]

    for total_precipitation in range(buffer_start, buffer_end + 1, buffer_step):
        print(f" Creating NetCDF for {base_name} with total precipitation = {total_precipitation} mm")

        nc_path = os.path.join(output_dir, f"{base_name}_{total_precipitation}.nc")

        precipitation_netcdf(dem_file_path, total_precipitation, nc_path, frequency_per_hour)

        print(f" Created NetCDF: {nc_path}")

#######################################################################################################

import os
from DEM_processing import precipitation_netcdf

def create_precip_automatic_camp_v2(base_name, dem_file_path, buffer_start, buffer_end, buffer_step, output_dir):
    """
    Generate NetCDF files for different precipitation intensities based on a single DEM file.
    Adds 15 minutes dry tail to the rainfall distribution at the end.
    Saves output to the given directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Frequency in mm/h per 5-minute step:
    frequency_core = [0.0, 9.30, 17.81, 18.70, 14.37, 11.13, 7.43, 5.28, 4.47, 3.41, 3.01, 2.68, 2.41]

    # Build full frequency distribution (18 steps = 90 minutes at 5-minute steps)
    frequency_per_hour = (
        frequency_core +      # 60 min rainfall (13 values)
        [0.0, 0.0, 0.0]  # 15 min dry tail
    )

    for total_precipitation in range(buffer_start, buffer_end + 1, buffer_step):
        print(f"Creating NetCDF for {base_name} with total precipitation = {total_precipitation} mm")

        nc_path = os.path.join(output_dir, f"{base_name}_{total_precipitation}.nc")

        precipitation_netcdf(dem_file_path, total_precipitation, nc_path, frequency_per_hour)

        print(f"Created NetCDF: {nc_path}")