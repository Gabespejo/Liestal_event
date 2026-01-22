import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd


def snap_bounds_to_grid(bounds, base=1000, mode="out"):
    """
    Snap bounds to nearest multiple of base (e.g. 1000 m).

    mode="out" → expand outward to nearest grid lines
    mode="in"  → shrink inward to nearest grid lines
    """
    minx, miny, maxx, maxy = bounds

    if mode == "out":
        minx = (minx // base) * base
        miny = (miny // base) * base
        maxx = ((maxx + base - 1) // base) * base  # ceil
        maxy = ((maxy + base - 1) // base) * base
    elif mode == "in":
        minx = ((minx + base - 1) // base) * base  # ceil
        miny = ((miny + base - 1) // base) * base
        maxx = (maxx // base) * base               # floor
        maxy = (maxy // base) * base
    else:
        raise ValueError("mode must be 'out' or 'in'")

    return float(minx), float(miny), float(maxx), float(maxy)


def make_square_bounds(bounds):
    """
    Convert a rectangle bounds (minx,miny,maxx,maxy) to a square bounds,
    by expanding the shorter side outward (centered).
    """
    minx, miny, maxx, maxy = bounds
    w = maxx - minx
    h = maxy - miny

    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    side = max(w, h)

    half = side / 2.0
    sq = (cx - half, cy - half, cx + half, cy + half)
    return sq


def crop_dem_to_Combiprecip_1km(
    full_dem: str,
    catchment_shp: str,
    id_field: str,
    id_value,
    output_dem: str,
    snap_res: int = 1000,
    mode: str = "out",
    make_square: bool = True,
    pad_m: float = 0.0,
    enforce_epsg2056: bool = False,
):
    """
    Crops a full-domain DEM to a square extent around a catchment polygon,
    and snaps the crop bounds to a 1 km grid (snap_res).

    - full_dem: path to your big DEM (GeoTIFF, etc.)
    - catchment_shp: shapefile/GeoPackage path with catchment polygons
    - id_field/id_value: which polygon to select
    - output_dem: where to save the cropped DEM
    - snap_res: 1000 for 1 km NetCDF compatibility
    - mode: 'out' usually (so polygon is fully included)
    - make_square: True to force quadratic domain
    - pad_m: optional padding (meters) before making square+snapping
    """

    if mode not in ("in", "out"):
        raise ValueError("mode must be 'in' or 'out'")

    os.makedirs(os.path.dirname(os.path.abspath(output_dem)), exist_ok=True)

    # --- Open DEM to get CRS + extent
    with rasterio.open(full_dem) as dem:
        dem_crs = dem.crs
        if dem_crs is None:
            raise ValueError("full_dem CRS is None. Please define CRS first.")

        if enforce_epsg2056:
            epsg = dem_crs.to_epsg()
            if epsg != 2056:
                raise ValueError(f"Expected EPSG:2056 but DEM is EPSG:{epsg}")

        dem_bounds = dem.bounds

        # --- Read catchment polygons
        gdf = gpd.read_file(catchment_shp)
        if id_field not in gdf.columns:
            raise ValueError(f"'{id_field}' not found in catchment file columns: {list(gdf.columns)}")

        sel = gdf[gdf[id_field] == id_value]
        if sel.empty:
            raise ValueError(f"No polygon found where {id_field} == {id_value}")

        # Reproject polygon to DEM CRS if needed
        if sel.crs is None:
            raise ValueError("Catchment file CRS is None. Define it first in QGIS or using geopandas.")
        if sel.crs != dem_crs:
            sel = sel.to_crs(dem_crs)

        geom = sel.geometry.unary_union
        minx, miny, maxx, maxy = geom.bounds

        # optional padding in meters
        if pad_m and pad_m > 0:
            minx -= pad_m
            miny -= pad_m
            maxx += pad_m
            maxy += pad_m

        bounds = (minx, miny, maxx, maxy)

        # make quadratic/square
        if make_square:
            bounds = make_square_bounds(bounds)

        # snap to 1 km grid so it aligns with your 1 km NetCDF cut
        snapped = snap_bounds_to_grid(bounds, base=snap_res, mode=mode)

        # Ensure snapped bounds stay within DEM extent (important)
        if (snapped[0] < dem_bounds.left or snapped[1] < dem_bounds.bottom or
            snapped[2] > dem_bounds.right or snapped[3] > dem_bounds.top):
            raise ValueError(
                "Snapped square bounds exceed full_dem extent.\n"
                f"Snapped: {snapped}\nDEM: {(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)}\n"
                "Use pad_m smaller, or use mode='in', or provide a bigger full_dem."
            )

        # Create window and crop
        win = from_bounds(*snapped, transform=dem.transform)
        win = win.round_offsets().round_lengths()

        data = dem.read(1, window=win)
        out_transform = dem.window_transform(win)

        profile = dem.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=out_transform,
            compress="lzw"
        )

    # Write cropped DEM
    with rasterio.open(output_dem, "w", **profile) as dst:
        dst.write(data, 1)

    print("✔ Cropped DEM written")
    print(f"✔ full_dem:      {full_dem}")
    print(f"✔ catchment:     {catchment_shp}")
    print(f"✔ selected:      {id_field} = {id_value}")
    print(f"✔ make_square:   {make_square}")
    print(f"✔ snap_res:      {snap_res} m  (NetCDF grid compatibility)")
    print(f"✔ mode:          {mode}")
    print(f"✔ snapped bounds:{snapped}")
    print(f"✔ output_dem:    {output_dem}")
    print(f"✔ out shape:     {data.shape} (rows, cols)")

    return snapped

##############################################################################################

import rasterio
import numpy as np

def convert_tif_to_asc(dem_tif, output_asc, desired_nodata_value=-9999):
    """
    Convert a GeoTIFF file to an Esri ASCII raster file and save CRS in a .prj file.

    Parameters:
    - dem_tif (str): Path to the input GeoTIFF file.
    - output_asc (str): Path to save the output Esri ASCII raster file.
    - desired_nodata_value (int/float): The NODATA value to replace any NaN or existing NODATA value.
    """
    # Open the .tif file
    with rasterio.open(dem_tif) as src:
        # Read the data and metadata
        data_tif = src.read(1)
        transform = src.transform
        crs = src.crs  # Preserve CRS
        original_nodata_value = src.nodata if src.nodata is not None else desired_nodata_value

        # Extract dimensions and transform properties
        ncols = src.width
        nrows = src.height
        xllcorner = transform[2]
        yllcorner = transform[5] - (nrows * abs(transform[4]))
        cellsize = transform[0]

    # Write the Esri ASCII raster file
    with open(output_asc, 'w') as asc_file:
        asc_file.write(f"ncols         {ncols}\n")
        asc_file.write(f"nrows         {nrows}\n")
        asc_file.write(f"xllcorner     {xllcorner}\n")
        asc_file.write(f"yllcorner     {yllcorner}\n")
        asc_file.write(f"cellsize      {cellsize}\n")
        asc_file.write(f"NODATA_value  {desired_nodata_value}\n")

        # Replace NaNs and original nodata values
        data_tif = np.where(np.isnan(data_tif) | (data_tif == original_nodata_value), desired_nodata_value, data_tif)
        np.savetxt(asc_file, data_tif, fmt="%.6f", delimiter=" ")

    # Save CRS to .prj file (if CRS exists)
    if crs:
        prj_file = output_asc.replace(".asc", ".prj")
        with open(prj_file, "w") as f:
            f.write(crs.to_wkt())  # Write CRS in WKT format

    # Print metadata
    print(f"Metadata:\n"
          f"ncols: {ncols}\n"
          f"nrows: {nrows}\n"
          f"xllcorner: {xllcorner}\n"
          f"yllcorner: {yllcorner}\n"
          f"cellsize: {cellsize}\n"
          f"NODATA_value: {desired_nodata_value}\n"
          f"CRS saved to: {prj_file if crs else 'No CRS found'}")

    print(f"Conversion complete. ASCII file saved to {output_asc}")


##################################################################################################

import rasterio
import os

def rename_file_extension(input_file_path, new_extension=".n", remove_original=False):
    """
    Rename an ASCII raster (.asc) file to a new extension by copying the contents
    and preserving metadata.

    Parameters:
        input_file_path (str): Path to the original .asc file.
        new_extension (str): New extension to apply (e.g., ".dem", ".n").
        remove_original (bool): If True, deletes the original file.

    Returns:
        str: Path to the newly created file with the new extension.
    """
    if not input_file_path.lower().endswith('.asc'):
        raise ValueError("Input file must have a .asc extension.")

    # Create new file path
    new_file_path = os.path.splitext(input_file_path)[0] + new_extension

    # Open and copy the raster
    with rasterio.open(input_file_path) as src:
        profile = src.profile
        profile.update(driver='AAIGrid')  # Keep ASCII format

        with rasterio.open(new_file_path, 'w', **profile) as dst:
            dst.write(src.read(1), 1)

    # Optionally remove original
    if remove_original:
        os.remove(input_file_path)

    print(f" File renamed to: {new_file_path}")
    return new_file_path
#####################################################################################################
########################### crop Combiprecip based on the tif- the same extent- in case the tif #####
###########################has been cut before snapped for the 1km of the combiprecip###############
####################################################################################################

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

    IMPORTANT: time is written like the "working" function:
      - time = 0..N-1
      - units = "1"
    """

    # Strip inputs (protect against accidental spaces from CLI)
    orig_nc = orig_nc.strip()
    dem_file = dem_file.strip()
    output_nc = output_nc.strip()

    # 1) Load bounds from *_bounds.txt (more robust than replace)
    root, _ = os.path.splitext(dem_file)
    bounds_txt = root + "_bounds.txt"

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
    da = ds_sel[var_name].transpose(time_name, y_name, x_name)
    data = np.nan_to_num(da.values.astype(np.float32), nan=0.0)

    x = da.coords[x_name].values.astype(np.float32)
    y = da.coords[y_name].values.astype(np.float32)
    nt, ny, nx = data.shape

    # 5) Write NetCDF
    os.makedirs(os.path.dirname(os.path.abspath(output_nc)), exist_ok=True)
    nc = Dataset(output_nc, "w")

    nc.createDimension("time", nt)
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)

    # ---- TIME: SAME AS YOUR WORKING FUNCTION ----
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

    nc.description = "Cropped deterministic CPC rainfall (snapped bounds; time=0..N-1)"
    nc.history     = f"Created on {date.today().isoformat()}"
    nc.source      = "CPC deterministic forecast cropped to DEM bounds (snapped)"

    nc.close()
    ds.close()

    print(f"✔ Saved: {output_nc}")
    print("✔ time written as float64: 0.0..N-1 (units=1)")


################################################################################################




