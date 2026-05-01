## Download DEM files
import os
import pandas as pd
import requests

def download_from_csvfile(csv_file: str, output_dir: str, url_column_index: int = 0):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file with explicit encoding and no header
    df = pd.read_csv(csv_file, encoding='utf-8-sig', header=None)

    # Extract links from the specified column
    links = df.iloc[:, url_column_index]

    # Print unique URLs
    print("Unique URLs in CSV:")
    unique_links = set(links)
    for link in unique_links:
        print(f"'{link}'")
    print(f"Total unique URLs found: {len(unique_links)}")

    # Print CSV content and columns
    print("CSV Content Preview:")
    print(df.head())
    print("CSV Columns:", df.columns)
    print("Total Rows in CSV:", len(df))

    # Check for empty or corrupted rows
    empty_rows = df[df.isnull().any(axis=1)]
    if not empty_rows.empty:
        print("Found empty or corrupted rows:")
        print(empty_rows)
    else:
        print("No empty or corrupted rows found.")

    # Print length of each URL
    for link in links:
        print(f"URL: '{link}' (Length: {len(link)})")

    # Track download status
    download_count = 0

    # Loop through the URLs and download each file
    for i, link in enumerate(links):
        link = str(link).strip().replace('\ufeff', '')
        print(f"Processing URL: '{link}' (Length: {len(link)})")

        if link.startswith("http"):
            try:
                print(f"Downloading {link}...")
                response = requests.get(link, stream=True, timeout=120)
                print(f"HTTP Status for {link}: {response.status_code}")
                response.raise_for_status()

                # Extract filename from URL
                filename = os.path.basename(link)
                print(f"Extracted filename: {filename}")
                output_path = os.path.join(output_dir, filename)

                # Save the file
                with open(output_path, "wb") as output_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        output_file.write(chunk)

                download_count += 1
                print(f"Saved: {output_path}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {link}: {e}")
                print(f"Error details: {response.text}")
            except Exception as e:
                print(f"Unexpected error for {link}: {e}")
        else:
            print(f"Invalid URL skipped: {link}")

    print(f"All downloads complete. Files downloaded: {download_count}/{len(links)}")

#### FIND RASTER FILES OF THE AREA OF INTEREST 

import os
import glob
import shutil
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, box

def extract_relevant_rasters(
    catchment_shapefile,
    location_csv,
    dem_folder,
    output_folder,
    location_id,
    campground_geojson,
    buffer_distance=100,
    crs="EPSG:2056"
):
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    gdf_catchments = gpd.read_file(catchment_shapefile)
    gdf_campgrounds = gpd.read_file(campground_geojson)
    df_location = pd.read_csv(location_csv)

    # Ensure CRS
    if gdf_catchments.crs is None:
        gdf_catchments.set_crs(crs, inplace=True)
    if gdf_campgrounds.crs is None:
        gdf_campgrounds.set_crs(crs, inplace=True)

    gdf_catchments = gdf_catchments.to_crs(crs)
    gdf_campgrounds = gdf_campgrounds.to_crs(crs)

    # Get location point from CSV
    if location_id not in df_location["ID"].values:
        raise ValueError(f"Location ID {location_id} not found in CSV.")
    
    location_point = Point(
        df_location[df_location["ID"] == location_id]["East_X"].values[0],
        df_location[df_location["ID"] == location_id]["North_Y"].values[0]
    )

    # Find the campground polygon that contains this point
    selected_camp = gdf_campgrounds[gdf_campgrounds.contains(location_point)]
    if selected_camp.empty:
        raise ValueError(f"No campground polygon contains the location point for ID {location_id}.")

    camp_geom = selected_camp.geometry.iloc[0]

    # Find all catchments that intersect the campground polygon
    intersecting_catchments = gdf_catchments[gdf_catchments.intersects(camp_geom)]
    if intersecting_catchments.empty:
        raise ValueError("No catchments intersect the campground polygon.")

    # Merge and buffer the intersecting catchments
    merged_geom = intersecting_catchments.unary_union.buffer(buffer_distance)

    # Final area to check for raster intersections
    combined_area = merged_geom

    # Get bounding box of the combined area
    minx, miny, maxx, maxy = combined_area.bounds

    # Make bounding box square
    x_diff, y_diff = maxx - minx, maxy - miny
    if x_diff > y_diff:
        maxy += x_diff - y_diff
    else:
        maxx += y_diff - x_diff

    quadratic_bbox = box(minx, miny, maxx, maxy)

    # Find raster tiles intersecting the square bounding box
    relevant_rasters = []
    for raster_path in glob.glob(os.path.join(dem_folder, "*.tif")):
        with rasterio.open(raster_path) as src:
            raster_bounds = src.bounds
            raster_geom = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
            if raster_geom.intersects(quadratic_bbox):
                relevant_rasters.append(raster_path)

    if not relevant_rasters:
        raise ValueError("No relevant rasters found intersecting the combined campground area.")

    # Copy raster files
    for raster in relevant_rasters:
        print(f" Copying: {os.path.basename(raster)}")
        shutil.copy(raster, output_folder)

    print(f" Relevant .tif files saved to {output_folder}")
############################################################################################

import os
import shutil
import pandas as pd
import glob
from shapely.geometry import box
import rasterio

#### This one is for Liestal_2m that we are using 

def snap_down(val, base=1000):
    return base * (val // base)


def extract_auto_matched_rasters(
    location_csv,
    dem_folder,
    output_folder,
    location_id,
    target_width=2000,  # grid cells
    target_height=2000,  # grid cells
    resolution=2.0  # meters
):
    """
    Automatically extract DEM tiles intersecting a bounding box that aligns to a 2 m grid and produces:
    - grid: 2000 × 2000
    - extent: 4000 m × 4000 m
    - bottom-left snapped automatically to nearest 1000 m grid
    """

    os.makedirs(output_folder, exist_ok=True)

    # Load location
    df = pd.read_csv(location_csv)
    row = df[df["ID"] == location_id]
    if row.empty:
        raise ValueError(f"Location ID {location_id} not found in CSV.")

    center_x = row.iloc[0]["East_X"]
    center_y = row.iloc[0]["North_Y"]

    half_extent = (target_width * resolution) / 2

    # Snap bottom-left corner of bbox to nearest 1000 m grid
    minx = snap_down(center_x - half_extent, base=1000)
    miny = snap_down(center_y - half_extent, base=1000)
    maxx = minx + (target_width * resolution)
    maxy = miny + (target_height * resolution)

    bbox = box(minx, miny, maxx, maxy)

    print(" Automatically snapped bounding box:")
    print(f"  X: {minx} → {maxx}")
    print(f"  Y: {miny} → {maxy}")
    print(f"  Grid: {target_width} × {target_height} at {resolution} m")

    # Find intersecting DEM tiles
    relevant_rasters = []
    for tif in glob.glob(os.path.join(dem_folder, "*.tif")):
        with rasterio.open(tif) as src:
            bounds = src.bounds
            tile_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            if tile_geom.intersects(bbox):
                relevant_rasters.append(tif)

    if not relevant_rasters:
        raise RuntimeError(" No DEM tiles found intersecting the generated area.")

    for raster in relevant_rasters:
        shutil.copy(raster, output_folder)
        print(f" Copied: {os.path.basename(raster)}")

    print(f" {len(relevant_rasters)} raster(s) saved to: {output_folder}")
    return minx, miny, maxx, maxy

#############################################################################################
###################### RESHAPE TO MATCH COMBIPRECIP IF I cropped it manually in Qgis ########
###########################################################################################
import os
import rasterio
from rasterio.windows import from_bounds

def snap_bounds_to_grid(bounds, base=1000, mode="in"):
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
        minx = ((minx + base - 1) // base) * base  # ceil up
        miny = ((miny + base - 1) // base) * base
        maxx = (maxx // base) * base               # floor down
        maxy = (maxy // base) * base
    else:
        raise ValueError("mode must be 'out' or 'in'")

    return float(minx), float(miny), float(maxx), float(maxy)


def resnap_dem_to_combiprecip(
    input_dem: str,
    output_dem: str,
    snap_res: int = 1000,
    mode: str = "out",
    full_dem: str | None = None,
    resampling: str = "bilinear",
    enforce_epsg2056: bool = False,
):
    """
    Robust for ANY resolution (2m / 1m / 0.5m / ...)

    Output extent = snapped bounds around input_dem (aligned to 1 km grid).
    Values:
      - start from full_dem (base) within snapped extent
      - overwrite with input_dem wherever input_dem has data

    No NODATA padding is introduced (because full_dem supplies values).

    Notes:
    - CRS is not forced, but full_dem CRS must match input_dem CRS.
    - If enforce_epsg2056=True, both must be EPSG:2056 or it raises.
    """

    if full_dem is None:
        raise ValueError("full_dem must be provided (base DEM for filling).")

    if mode not in ("in", "out"):
        raise ValueError("mode must be 'in' or 'out'")

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    rs = resampling_map.get(resampling, Resampling.bilinear)

    os.makedirs(os.path.dirname(os.path.abspath(output_dem)), exist_ok=True)

    # --- 1) read input (small) DEM and compute snapped bounds ---
    with rasterio.open(input_dem) as s:
        if s.crs is None:
            raise ValueError("input_dem CRS is None. Define CRS first.")
        small_crs = s.crs
        small_transform = s.transform
        small_nodata = s.nodata
        small_data = s.read(1).astype(np.float32)

        # snapped extent based on SMALL DEM bounds
        snapped = snap_bounds_to_grid(s.bounds, base=snap_res, mode=mode)

    # --- 2) crop full_dem to snapped bounds (defines output grid/extent) ---
    with rasterio.open(full_dem) as f:
        if f.crs != small_crs:
            raise ValueError(f"CRS mismatch: input_dem={small_crs}, full_dem={f.crs}")

        if enforce_epsg2056:
            try:
                if int(small_crs.to_epsg() or -1) != 2056:
                    raise ValueError(f"CRS must be EPSG:2056, got {small_crs}")
            except Exception:
                raise ValueError(f"CRS must be EPSG:2056, got {small_crs}")

        # ensure snapped bounds lie inside full_dem
        fb = f.bounds
        if (snapped[0] < fb.left or snapped[1] < fb.bottom or
            snapped[2] > fb.right or snapped[3] > fb.top):
            raise ValueError(
                "Snapped bounds exceed full_dem extent.\n"
                "Use a larger full_dem OR set mode='in' to shrink."
            )

        win = from_bounds(*snapped, transform=f.transform)
        win = win.round_offsets().round_lengths()

        # base is full_dem cropped to snapped window (so output is NOT huge)
        base_crop = f.read(1, window=win).astype(np.float32)
        out_transform = f.window_transform(win)

        out_profile = f.profile.copy()
        out_profile.update(
            height=base_crop.shape[0],
            width=base_crop.shape[1],
            transform=out_transform,
            dtype="float32",
            compress="lzw",
        )

    # --- 3) reproject input_dem onto output grid and overwrite base ---
    tmp = np.full(base_crop.shape, np.nan, dtype=np.float32)

    reproject(
        source=small_data,
        destination=tmp,
        src_transform=small_transform,
        src_crs=small_crs,
        src_nodata=small_nodata,
        dst_transform=out_transform,
        dst_crs=small_crs,
        dst_nodata=np.nan,
        resampling=rs,
    )

    mask = ~np.isnan(tmp)
    base_crop[mask] = tmp[mask]

    # --- 4) write output ---
    with rasterio.open(output_dem, "w", **out_profile) as dst:
        dst.write(base_crop, 1)

    print("✔ DEM prepared (cropped to snapped extent + filled from full_dem)")
    print(f"✔ input_dem:   {input_dem}")
    print(f"✔ full_dem:    {full_dem}")
    print(f"✔ output_dem:  {output_dem}")
    print(f"✔ snapped:     {snapped}")
    print(f"✔ out shape:   {base_crop.shape}  (rows, cols)")
    print(f"✔ out res:     {out_profile['transform'][0]} m")

    return snapped


##############################################################################################

import os
import shutil
import pandas as pd
import glob
from shapely.geometry import box
import rasterio

def extract_relevant_rasters_v1(
    location_csv,
    dem_folder,
    output_folder,
    location_id,
    square_extent=4000,  # in meters (e.g., 10000 to create a 10x10 km area)
    crs="EPSG:2056"
):
    """
    Extract and copy DEM raster tiles that intersect with an exact square extent
    (no snapping) around a point location from a CSV file.

    - Bounding box will be strictly square_extent × square_extent.
    - Centered on the selected location point.
    - Copies all .tif tiles from `dem_folder` that intersect the box to `output_folder`.
    """

    os.makedirs(output_folder, exist_ok=True)

    # Load CSV with location points
    df_location = pd.read_csv(location_csv)
    if not {"ID", "East_X", "North_Y"}.issubset(df_location.columns):
        raise ValueError("CSV must contain 'ID', 'East_X', and 'North_Y' columns.")

    # Get the point by ID
    selected_row = df_location[df_location["ID"] == location_id]
    if selected_row.empty:
        raise ValueError(f"Location ID {location_id} not found in CSV.")

    center_x = selected_row.iloc[0]["East_X"]
    center_y = selected_row.iloc[0]["North_Y"]
    half_extent = square_extent / 2

    # Exact bounding box (no snapping)
    minx = center_x - half_extent
    maxx = center_x + half_extent
    miny = center_y - half_extent
    maxy = center_y + half_extent

    bbox = box(minx, miny, maxx, maxy)
    print(f" Using exact extent: {maxx - minx}m × {maxy - miny}m")
    print(f" Exact bounding box: X: {minx} → {maxx}, Y: {miny} → {maxy}")

    # Find DEM .tif files that intersect this bbox
    relevant_rasters = []
    for raster_path in glob.glob(os.path.join(dem_folder, "*.tif")):
        with rasterio.open(raster_path) as src:
            raster_bounds = src.bounds
            raster_geom = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
            if raster_geom.intersects(bbox):
                relevant_rasters.append(raster_path)

    if not relevant_rasters:
        raise ValueError("No .tif files found intersecting the specified area.")

    # Copy files to output
    for raster in relevant_rasters:
        print(f" Copying: {os.path.basename(raster)}")
        shutil.copy(raster, output_folder)

    print(f" Relevant DEM tiles saved to: {output_folder}")


##############################################################################################

import os
import shutil
import pandas as pd
import glob
from shapely.geometry import box
import rasterio

def snap_to_grid(val, step=2, direction='down'):
    """
    Snap a coordinate to the nearest lower or upper multiple of `step`.
    Useful for aligning bounding boxes to 2 m DEM grid.
    """
    if direction == 'down':
        return step * (val // step)
    else:
        return step * ((val + step - 1) // step)

def extract_relevant_rasters_v2(
    location_csv,
    dem_folder,
    output_folder,
    location_id,
    square_extent=4000,  # in meters (e.g., 20000 for 20 km × 20 km)
    crs="EPSG:2056"
):
    """
    Extract and copy DEM raster tiles that intersect with a snapped square extent
    around a point location from a CSV file.

    - Bounding box is snapped to a 2 m grid.
    - `square_extent` must be a multiple of 2 for LISFLOOD compatibility.
    - Copies all .tif tiles from `dem_folder` that intersect the square to `output_folder`.
    """

    if square_extent % 2 != 0:
        raise ValueError(" 'square_extent' must be a multiple of 2 to align with 2 m grid.")

    os.makedirs(output_folder, exist_ok=True)

    # Load CSV with location points
    df_location = pd.read_csv(location_csv)
    required_cols = {"ID", "East_X", "North_Y"}
    if not required_cols.issubset(df_location.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Get the point by ID
    selected_row = df_location[df_location["ID"] == location_id]
    if selected_row.empty:
        raise ValueError(f"Location ID {location_id} not found in CSV.")

    center_x = selected_row.iloc[0]["East_X"]
    center_y = selected_row.iloc[0]["North_Y"]
    half_extent = square_extent / 2

    # Snap bounding box to 2 m grid
    minx = snap_to_grid(center_x - half_extent, step=2, direction='down')
    maxx = minx + square_extent
    miny = snap_to_grid(center_y - half_extent, step=2, direction='down')
    maxy = miny + square_extent

    bbox = box(minx, miny, maxx, maxy)
    print(f" Snapped extent: {maxx - minx} m × {maxy - miny} m")
    print(f" Bounding box: X: {minx} → {maxx}, Y: {miny} → {maxy}")

    # Find DEM .tif files that intersect this bbox
    relevant_rasters = []
    for raster_path in glob.glob(os.path.join(dem_folder, "*.tif")):
        with rasterio.open(raster_path) as src:
            raster_bounds = src.bounds
            raster_geom = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
            if raster_geom.intersects(bbox):
                relevant_rasters.append(raster_path)

    if not relevant_rasters:
        raise ValueError(" No .tif files found intersecting the snapped area.")

    # Copy matching raster files
    for raster in relevant_rasters:
        print(f" Copying: {os.path.basename(raster)}")
        shutil.copy(raster, output_folder)

    print(f" Relevant DEM tiles saved to: {output_folder}")
    
    # Optional: return bounding box for further processing
    return minx, miny, maxx, maxy

##############################################################################################
################################### download flex for all resolution #########################
##########################################################################################

import os, glob, shutil, math
import pandas as pd
import rasterio
from shapely.geometry import box

def _is_multiple(value, step, tol=1e-9):
    """Return True if value is an integer multiple of step (with tolerance)."""
    q = value / step
    return abs(q - round(q)) < tol

def _snap(value, step, how="down"):
    """Snap value to a grid of spacing `step`."""
    q = value / step
    if how == "down":
        return math.floor(q) * step
    elif how == "up":
        return math.ceil(q) * step
    else:
        return round(q) * step  # nearest

def extract_relevant_rasters_flex(
    location_csv: str,
    dem_folder: str,
    output_folder: str,
    location_id,
    square_extent: float = 4000.0,   # total width/height of the square (in DEM CRS units, meters for EPSG:2056)
    grid_res: float | None = 2.0,    # DEM cell size; set to 0.5 for 50 cm. If None, auto-detects from a tile.
    crs: str = "EPSG:2056"
):
    """
    Extract and copy DEM GeoTIFF tiles that intersect a snapped square around a CSV point.

    - The square is centered on the point `ID == location_id`.
    - The square bounds are snapped to the DEM grid of size `grid_res`.
    - `square_extent` must be a multiple of `grid_res` (LISFLOOD-friendly).
    - All *.tif files in `dem_folder` intersecting the snapped square are copied to `output_folder`.

    Returns (minx, miny, maxx, maxy) of the snapped square.
    """

    os.makedirs(output_folder, exist_ok=True)

    # --- Read locations CSV
    df = pd.read_csv(location_csv)
    required = {"ID", "East_X", "North_Y"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    row = df[df["ID"] == location_id]
    if row.empty:
        raise ValueError(f"Location ID {location_id} not found in {location_csv}")

    cx = float(row.iloc[0]["East_X"])
    cy = float(row.iloc[0]["North_Y"])

    # --- Auto-detect grid_res if requested
    if grid_res is None:
        sample_tiles = glob.glob(os.path.join(dem_folder, "*.tif"))
        if not sample_tiles:
            raise ValueError(f"No .tif files found in {dem_folder} to auto-detect resolution.")
        with rasterio.open(sample_tiles[0]) as src:
            # assumes square pixels; transform.a is pixel width
            grid_res = float(abs(src.transform.a))
        print(f"ℹ Detected DEM resolution: {grid_res} (units of CRS)")

    # --- Validate extent vs grid
    if not _is_multiple(square_extent, grid_res):
        raise ValueError(
            f"'square_extent' ({square_extent}) must be a multiple of grid_res ({grid_res})."
        )

    half = square_extent / 2.0

    # --- Snap to grid
    minx = _snap(cx - half, grid_res, "down")
    miny = _snap(cy - half, grid_res, "down")
    maxx = minx + square_extent
    maxy = miny + square_extent

    bbox = box(minx, miny, maxx, maxy)
    print(f" Snapped extent: {maxx - minx:.3f} × {maxy - miny:.3f} (grid {grid_res})")
    print(f"   Bounds: X [{minx:.3f}, {maxx:.3f}], Y [{miny:.3f}, {maxy:.3f}]")

    # --- Find intersecting tiles
    relevant = []
    for tif in glob.glob(os.path.join(dem_folder, "*.tif")):
        try:
            with rasterio.open(tif) as src:
                b = src.bounds
                rgeom = box(b.left, b.bottom, b.right, b.top)
                if rgeom.intersects(bbox):
                    relevant.append(tif)
        except Exception as e:
            print(f"⚠ Skipping {tif}: {e}")

    if not relevant:
        raise ValueError("No .tif tiles intersect the snapped area.")

    # --- Copy tiles
    for tif in relevant:
        dst = os.path.join(output_folder, os.path.basename(tif))
        print(f"→ Copying {os.path.basename(tif)}")
        shutil.copy(tif, dst)

    print(f" {len(relevant)} tile(s) copied to: {output_folder}")
    return minx, miny, maxx, maxy



# Re-import libraries after code execution environment reset
import os
import glob
import shutil
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, box

def extract_rasters_near_campground(
    location_csv,
    campground_geojson,
    dem_folder,
    output_folder,
    location_id,
    half_extent=2000,  # 2 km in each direction = 4 km square
    crs="EPSG:2056"
):
    os.makedirs(output_folder, exist_ok=True)

    # Load campground polygons and location CSV
    gdf_campgrounds = gpd.read_file(campground_geojson)
    df_location = pd.read_csv(location_csv)

    # Ensure CRS is set
    if gdf_campgrounds.crs is None:
        gdf_campgrounds.set_crs(crs, inplace=True)
    gdf_campgrounds = gdf_campgrounds.to_crs(crs)

    # Get point location for the specified ID
    if location_id not in df_location["ID"].values:
        raise ValueError(f"Location ID {location_id} not found in CSV.")
    
    x = df_location[df_location["ID"] == location_id]["East_X"].values[0]
    y = df_location[df_location["ID"] == location_id]["North_Y"].values[0]
    location_point = Point(x, y)

    # Find campground polygon containing the point
    selected_camp = gdf_campgrounds[gdf_campgrounds.contains(location_point)]
    if selected_camp.empty:
        raise ValueError(f"No campground polygon contains the point for ID {location_id}.")
    
    # Create square bounding box around the point (4x4 km, centered)
    minx, miny = x - half_extent, y - half_extent
    maxx, maxy = x + half_extent, y + half_extent
    square_bbox = box(minx, miny, maxx, maxy)

    # Find intersecting rasters
    relevant_rasters = []
    for raster_path in glob.glob(os.path.join(dem_folder, "*.tif")):
        with rasterio.open(raster_path) as src:
            raster_bounds = src.bounds
            raster_geom = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
            if raster_geom.intersects(square_bbox):
                relevant_rasters.append(raster_path)

    if not relevant_rasters:
        raise ValueError("No relevant rasters found intersecting the 4 km square area around the campground.")

    # Copy the matching raster files to the output folder
    for raster in relevant_rasters:
        print(f"Copying: {os.path.basename(raster)}")
        shutil.copy(raster, output_folder)

    print(f"Extracted relevant rasters saved to: {output_folder}")

####MERGED THE RASTER FILES OF AREA OF INTEREST

import os
import glob
from osgeo import gdal


def merge_dem_rasters(
    input_folder,
    output_file,
    bounds=None,
    output_format="GTiff",
    data_type=gdal.GDT_Float32,
    xRes=2.0,
    yRes=2.0,
):
    """
    Merges all .tif files into a single DEM with fixed bounds and resolution.

    Parameters
    ----------
    xRes, yRes : pixel size (e.g., 0.5 for 50 cm; 2.0 for 2 m)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    relevant_rasters = glob.glob(os.path.join(input_folder, "*.tif"))

    if not relevant_rasters:
        raise ValueError("No relevant raster files found.")

    print("Merging with bounds:")
    print(f"  {bounds}")
    print(f"Pixel size: {xRes} x {yRes}")

    gdal.Warp(destNameOrDestDS=output_file,
            srcDSOrSrcDSTab=relevant_rasters,
            format=output_format,
            outputType=data_type,
            multithread=True,
            outputBounds=bounds if bounds else None,
            xRes=xRes,
            yRes=yRes,
            targetAlignedPixels=True
            )

    if os.path.exists(output_file):
        print(f" Merged DEM saved: {output_file}")
    else:
        print(" Merging failed.")

    
################################################################################################
######### per bounding as input ################################################################
import os
import glob
from osgeo import gdal

def merge_dem_rasters_binput(input_folder, output_file, bbox, output_format="GTiff", data_type=gdal.GDT_Float32):
    """
    Merges all .tif files in the specified input folder into a single DEM
    and crops to the provided bounding box.

    Parameters:
        input_folder (str): Folder containing .tif files to merge.
        output_file (str): Path to the output merged raster.
        bbox (tuple): Bounding box (xmin, ymin, xmax, ymax) in EPSG:2056.
        output_format (str): GDAL format (default: GTiff).
        data_type (gdal type): Output data type (default: gdal.GDT_Float32).
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Collect .tif files
    relevant_rasters = glob.glob(os.path.join(input_folder, "*.tif"))
    if not relevant_rasters:
        raise ValueError(" No .tif files found to merge.")
    print(f" Merging {len(relevant_rasters)} rasters...")

    # Merge and clip to bounding box
    gdal.Warp(
        destNameOrDestDS=output_file,
        srcDSOrSrcDSTab=relevant_rasters,
        format=output_format,
        outputType=data_type,
        outputBounds=bbox,  #  Crop here!
        multithread=True
    )

    if os.path.exists(output_file):
        print(f" Merged and cropped DEM saved to: {output_file}")
    else:
        print(" Merge failed.")

######################## resample DEM OR TIF FILE ###############################################

import os
from osgeo import gdal

def resample_raster_swisssurface3d(
    input_raster,
    output_raster,
    xRes=2.0,
    yRes=2.0,
    resampleAlg="max"
):
    """
    Resample raster to new resolution.

    Parameters
    ----------
    input_raster : path to input raster
    output_raster : path to output raster
    xRes, yRes : target resolution (e.g. 2.0 for 2 m)
    resampleAlg : resampling method ("max" recommended for DSM)
    """

    os.makedirs(os.path.dirname(output_raster), exist_ok=True)

    print(f"Resampling {input_raster}")
    print(f"Target resolution: {xRes} x {yRes}")
    print(f"Resampling method: {resampleAlg}")

    gdal.Warp(
        destNameOrDestDS=output_raster,
        srcDSOrSrcDSTab=input_raster,
        xRes=xRes,
        yRes=yRes,
        resampleAlg=resampleAlg,
        targetAlignedPixels=True,
        multithread=True
    )

    print(f"Done: {output_raster}")


##################################################################################################
###################### clip to Manning values bbox ##############################################
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from pyproj import CRS, Transformer
import os

def clip_raster_to_bbox(
    input_raster,
    output_raster,
    bbox,  # (minx, miny, maxx, maxy)
    bbox_crs=None  # Optional: if bbox is in a different CRS than the raster
):
    """
    Clips a raster to a given bounding box and saves it.

    Parameters:
        input_raster (str): Path to input .tif raster.
        output_raster (str): Path to save the clipped raster.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy).
        bbox_crs (str or rasterio CRS, optional): CRS of bbox (e.g., "EPSG:4326").
                                                  If different from raster CRS, will be reprojected.
    Returns:
        str: Path to the saved clipped raster.
    """
    minx, miny, maxx, maxy = bbox

    with rasterio.open(input_raster) as src:
        raster_crs = src.crs

        # If bbox CRS is different, reproject bbox to match raster
        if bbox_crs and CRS.from_user_input(bbox_crs) != raster_crs:
            print("🔄 Reprojecting bbox to match raster CRS...")
            transformer = Transformer.from_crs(bbox_crs, raster_crs, always_xy=True)
            minx, miny = transformer.transform(minx, miny)
            maxx, maxy = transformer.transform(maxx, maxy)
            print(f"📍 Reprojected BBox: ({minx}, {miny}, {maxx}, {maxy})")

        # Compute window and transform
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = src.window_transform(window)
        data = src.read(1, window=window)

        # Prepare metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform
        })

        # Save output
        os.makedirs(os.path.dirname(output_raster), exist_ok=True)
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(data, 1)

    print(f"✅ Clipped raster saved to: {output_raster}")
    return output_raster

######################################################################################################
import os
import pandas as pd
import requests

def download_swissimage_tif(csv_file, output_folder):
    """
    Downloads .tif files from URLs listed in a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file containing URLs.
        output_folder (str): Path to the folder where .tif files will be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Assume the links are in the first column
    links = df.iloc[:, 0]  # Modify column index if necessary

    # Loop through the URLs and download each file
    for i, link in enumerate(links):
        link = str(link).strip()  # Remove any leading/trailing whitespace
        if link.startswith("http"):  # Ensure it's a valid URL
            try:
                print(f"Downloading {link}...")
                response = requests.get(link, stream=True)
                response.raise_for_status()  # Raise an error for bad responses

                # Extract filename from URL
                filename = os.path.basename(link)
                output_path = os.path.join(output_folder, filename)

                # Save the file
                with open(output_path, "wb") as output_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        output_file.write(chunk)

                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed to download {link}: {e}")
        else:
            print(f"Invalid URL skipped: {link}")

    print("All downloads complete.")

###########

import os
from osgeo import gdal

def merge_swissimage_tif(tif_folder, output_file):
    """
    Merges all .tif files in a specified folder into a single .tif file.

    Parameters:
        tif_folder (str): Path to the folder containing .tif files.
        output_file (str): Path to save the merged .tif file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Find all .tif files in the folder
    tif_files = [os.path.join(tif_folder, f) for f in os.listdir(tif_folder) if f.endswith(".tif")]

    # Check if there are files to merge
    if len(tif_files) == 0:
        print("No .tif files found in the specified folder.")
        return
    else:
        print(f"Found {len(tif_files)} .tif files. Merging...")

    # Use gdal.Warp to merge the files
    try:
        gdal.Warp(output_file, tif_files, format="GTiff")
        print(f"Merging completed successfully. Merged file saved at: {output_file}")
    except Exception as e:
        print(f"Error during merging: {e}")

###################################################################################################
###### READ THE DEM FILES ######
import rasterio

def read_DEM(file_path):
    """
    Function to read metadata and the first band of a GeoTIFF raster file.
    
    Parameters:
    file_path (str): Path to the GeoTIFF raster file.
    
    Returns:
    dict: A dictionary containing metadata and the first band (2D numpy array).
    """
    with rasterio.open(file_path) as src:
        # Extract basic metadata
        metadata = {
            "Raster Width": src.width,
            "Raster Height": src.height,
            "Coordinate Reference System (CRS)": src.crs,
            "Number of Bands": src.count,
            "Bounds": src.bounds
        }

        # Read the first band (as a 2D numpy array)
        band1 = src.read(1)

    return metadata, band1

###########################################################################
####### RESAMPLE THE DEM OF RESOLUTION ####################################
###########################################################################
from osgeo import gdal

def DEM_resample_resolution(input_dem, output_dem, target_resolution, resample_method="average"):
    """
    Resamples a DEM to a specified resolution using the chosen resampling method.

    Parameters:
    - input_dem (str): Path to the input DEM file.
    - output_dem (str): Path to the output resampled DEM file.
    - target_resolution (float): Desired resolution in meters (e.g., 10).
    - resample_method (str): Resampling method ('average', 'bilinear', 'nearest', etc.). Default is 'average'.

    Notes:
    - 'average' works only with GDAL >= 3.3.
    - For elevation data, 'average' or 'bilinear' is preferred.
    """

    supported_methods = ['nearest', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average', 'mode', 'max', 'min', 'med', 'q1', 'q3']

    if resample_method not in supported_methods:
        raise ValueError(f" Unsupported resampling method: '{resample_method}'.\nChoose from: {supported_methods}")

    print(f"📐 Resampling DEM from original to {target_resolution} m resolution using '{resample_method}'...")

    resampled = gdal.Warp(
        destNameOrDestDS=output_dem,
        srcDSOrSrcDSTab=input_dem,
        format="GTiff",
        xRes=target_resolution,
        yRes=target_resolution,
        resampleAlg=resample_method,
        dstNodata=-9999,
        multithread=True,
        creationOptions=["COMPRESS=LZW"]
    )

    # Properly close and flush to disk
    resampled = None
    print(f" Resampled DEM saved to: {output_dem}")

#########################################################################
## CLIP THE DEM ONLY FORTHE CATCHMENT 
########################################################################

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import rasterio
from rasterio.mask import mask

def CLIP_DEM_CATCHMENT(
    geo_ezgg_shapefile,
    catchment_csv,
    dem_file,
    output_raster,
    chosen_id,
    buffer_distance=100
):
    """
    Clip DEM based on catchment that contains a point (from CSV) with optional buffer.
    
    Parameters:
    - geo_ezgg_shapefile (str): Path to the catchments shapefile.
    - catchment_csv (str): Path to the CSV file with ID, East_X, North_Y.
    - dem_file (str): Path to input DEM (GeoTIFF).
    - output_raster (str): Path to save the clipped raster.
    - chosen_id (int): ID from the CSV to locate point.
    - buffer_distance (int): Buffer distance in meters (default = 100).
    """

    # Load catchments and CSV
    gdf_catchments = gpd.read_file(geo_ezgg_shapefile)
    df_location = pd.read_csv(catchment_csv)

    # Filter the point from the CSV
    chosen_point = df_location[df_location['ID'] == chosen_id]
    if chosen_point.empty:
        raise ValueError(f" ID {chosen_id} not found in CSV.")

    # Create a point geometry from coordinates
    point_geom = Point(chosen_point["East_X"].iloc[0], chosen_point["North_Y"].iloc[0])
    gdf_point = gpd.GeoDataFrame(geometry=[point_geom], crs="EPSG:2056")  # update if needed

    # Open DEM and get CRS
    with rasterio.open(dem_file) as src:
        dem_crs = src.crs

        # Reproject catchments and point to match DEM
        gdf_catchments = gdf_catchments.to_crs(dem_crs)
        gdf_point = gdf_point.to_crs(dem_crs)

        # Get point geometry again
        point = gdf_point.geometry.iloc[0]

        # Find catchment containing the point
        selected_catchment = gdf_catchments[gdf_catchments.contains(point)]
        if selected_catchment.empty:
            raise ValueError(f" No catchment contains the point for ID {chosen_id}.")

        # Optional buffer
        selected_catchment_buffered = selected_catchment.copy()
        selected_catchment_buffered["geometry"] = selected_catchment_buffered.geometry.buffer(buffer_distance)

        # Clip DEM
        out_image, out_transform = mask(src, selected_catchment_buffered.geometry, crop=True)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    # Save output raster
    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Clipped DEM saved to: {output_raster}")

###################################################################################################################################

import geopandas as gpd
import rasterio
from shapely.geometry import mapping, box
from rasterio.mask import mask

def CLIP_DEM_by_campground_and_catchments(
    geo_ezgg_shapefile,
    campground_geojson,
    dem_file,
    output_raster,
    chosen_id
):
    """
    Clip DEM using the campground polygon and catchments that intersect it.
    Handles CRS and fixes geometries like QGIS.
    """

    # --- Load DEM and get CRS ---
    with rasterio.open(dem_file) as src:
        dem_crs = src.crs
        dem_bounds = src.bounds

        # --- Load and fix catchments ---
        gdf_catchments = gpd.read_file(geo_ezgg_shapefile)
        if gdf_catchments.crs is None:
            gdf_catchments.set_crs(dem_crs, inplace=True)
        gdf_catchments = gdf_catchments.to_crs(dem_crs)
        gdf_catchments["geometry"] = gdf_catchments.buffer(0)  # fix geometry

        # --- Load and fix campgrounds ---
        gdf_campgrounds = gpd.read_file(campground_geojson)
        if gdf_campgrounds.crs is None:
            gdf_campgrounds.set_crs(dem_crs, inplace=True)
        gdf_campgrounds = gdf_campgrounds.to_crs(dem_crs)
        gdf_campgrounds["geometry"] = gdf_campgrounds.buffer(0)  # fix geometry

        # --- Select campground polygon ---
        selected_camp = gdf_campgrounds[gdf_campgrounds["ID"] == chosen_id]
        if selected_camp.empty:
            raise ValueError(f" Campground ID {chosen_id} not found.")
        camp_geom = selected_camp.geometry.iloc[0]

        # --- Select intersecting catchments ---
        intersecting_catchments = gdf_catchments[gdf_catchments.intersects(camp_geom)]
        if intersecting_catchments.empty:
            raise ValueError(" No catchments intersect with the campground.")

        # --- Merge geometries: camp + catchments ---
        merged_geom = gpd.GeoSeries([camp_geom] + list(intersecting_catchments.geometry), crs=dem_crs).unary_union

        if merged_geom.is_empty:
            raise ValueError("Final merged geometry is empty after union.")

        if not merged_geom.intersects(box(*dem_bounds)):
            raise ValueError(" Geometry does not intersect DEM. Likely CRS or extent mismatch.")

        # --- Clip the DEM ---
        out_image, out_transform = mask(src, [mapping(merged_geom)], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    # --- Save the output raster ---
    with rasterio.open(output_raster, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f" DEM clipped and saved to: {output_raster}")

    #####################################################################

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import rasterio.transform  # <-- important!
import pandas as pd
import numpy as np

def geopackage_to_raster(arealstatistik, column_name, output_raster, resolution=100):
    """
    Converts a column from a GeoDataFrame into a raster GeoTIFF with the given resolution.

    Parameters:
    - arealstatistik (GeoDataFrame): GeoDataFrame with the column to rasterize.
    - column_name (str): Name of the column containing values to rasterize.
    - output_raster (str): File path to save the raster.
    - resolution (int or float): Pixel resolution in map units.
    """

    # Clean column names
    arealstatistik.columns = arealstatistik.columns.str.strip()

    # Check if column exists
    if column_name not in arealstatistik.columns:
        raise KeyError(f"The column '{column_name}' was not found in the GeoDataFrame.")
    print(f" Column '{column_name}' found.")

    # Ensure the values are numeric
    arealstatistik[column_name] = pd.to_numeric(arealstatistik[column_name], errors='coerce')

    # Check if CRS exists
    if arealstatistik.crs is None:
        raise ValueError("The GeoDataFrame has no CRS defined. Please set a valid CRS before rasterizing.")

    # Get total bounds
    minx, miny, maxx, maxy = arealstatistik.total_bounds

    # Ensure bounds are valid
    if maxx <= minx or maxy <= miny:
        raise ValueError("Invalid bounds: width or height would be zero or negative.")

    # Calculate raster width and height
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    if width == 0 or height == 0:
        raise ValueError("Calculated width or height is 0. Check resolution or geometry bounds.")

    # Create transform (now explicitly from rasterio.transform)
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    # Prepare shapes, skip NaN and invalid geometries
    shapes = [
        (geom, value)
        for geom, value in zip(arealstatistik.geometry, arealstatistik[column_name])
        if geom.is_valid and not np.isnan(value)
    ]

    if len(shapes) == 0:
        raise ValueError("No valid geometries with values to rasterize.")

    # Rasterize
    raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='float32'
    )

    # Write to GeoTIFF
    with rasterio.open(
        output_raster,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=arealstatistik.crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(raster, 1)

    print(f" Raster saved to: {output_raster}")

##########################################################################################


import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

def areal_to_demtarget_resampling(dem_file, arealstatistik_file, output_raster, resampling_method=Resampling.nearest):
    """
    Function to resample the Arealstatistik raster to match the resolution, extent, and CRS of the DEM raster,
    using a DEM mask and nearest-neighbor resampling (suitable for categorical data).

    Parameters:
    - dem_file (str): Path to the input DEM raster (GeoTIFF).
    - arealstatistik_file (str): Path to the Arealstatistik raster (GeoTIFF) to be resampled.
    - output_raster (str): Path to save the output clipped and resampled raster.
    - resampling_method (Resampling): The resampling method to be used (default is nearest).
    """

    # Step 1: Open the DEM raster
    with rasterio.open(dem_file) as dem_src:
        dem_data = dem_src.read(1)
        dem_mask = dem_data != dem_src.nodata
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        dem_shape = dem_src.shape

    # Step 2: Open the Arealstatistik raster
    with rasterio.open(arealstatistik_file) as arealstatistik_src:
        if arealstatistik_src.crs != dem_crs:
            raise ValueError("CRS mismatch between rasters. Reproject the Arealstatistik raster to match the DEM CRS.")

        # Reproject Arealstatistik to match DEM
        reprojected_arealstatistik = np.zeros(dem_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(arealstatistik_src, 1),
            destination=reprojected_arealstatistik,
            src_transform=arealstatistik_src.transform,
            src_crs=arealstatistik_src.crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            dst_shape=dem_shape,
            resampling=resampling_method
        )

    # Step 3: Apply DEM mask
    masked_arealstatistik = np.where(dem_mask, reprojected_arealstatistik, np.nan)

    # Step 4: Update metadata for output
    dem_meta.update({
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": np.nan
    })

    # Step 5: Write output raster
    with rasterio.open(output_raster, "w", **dem_meta) as dest:
        dest.write(masked_arealstatistik, 1)

    print(f"Clipped and resampled raster saved to {output_raster}")


##########################################################################################################
####CLIP FIRST THE AREALSTATISTIK FOR THE SECTOR OF THE CATCHMENT BUT IN THIS CASE IN QUADRATIC FORM #####
##########################################################################################################
##########################################################################################################
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from rasterio.windows import from_bounds

def clip_raster(catchment_shapefile, csv_file, raster_file, output_raster, catchment_id, buffer_distance):
    """
    Clips a raster using the bounding box of a selected catchment polygon with a specified buffer distance.
    
    Parameters:
        catchment_shapefile (str): Path to the catchment shapefile.
        csv_file (str): Path to the CSV file containing location points.
        raster_file (str): Path to the input raster file.
        output_raster (str): Path to save the clipped raster.
        catchment_id (int): ID of the catchment to select.
        buffer_distance (float): Buffer distance in meters to extend the bounding box.
    """
    # Step 1: Load the CSV file with location points
    df_location = pd.read_csv(csv_file)
    
    # Ensure required columns exist
    if not {"ID", "East_X", "North_Y"}.issubset(df_location.columns):
        raise ValueError("CSV file must contain 'ID', 'East_X', and 'North_Y' columns.")
    
    # Select the specific catchment row
    selected_row = df_location[df_location["ID"] == catchment_id]
    if selected_row.empty:
        raise ValueError(f"Catchment ID {catchment_id} not found in the CSV.")
    
    east_x, north_y = selected_row.iloc[0][["East_X", "North_Y"]]
    
    # Step 2: Load the Catchment Shapefile
    catchments = gpd.read_file(catchment_shapefile)
    
    # Convert the point to the same CRS as the shapefile
    point = gpd.GeoDataFrame(geometry=[Point(east_x, north_y)], crs="EPSG:2056")  # Assuming Swiss CH1903+
    point = point.to_crs(catchments.crs)
    
    # Step 3: Select the Polygon Containing the Point
    selected_catchment = catchments[catchments.contains(point.iloc[0].geometry)]
    if selected_catchment.empty:
        raise ValueError(f"No catchment polygon found for ID {catchment_id} at ({east_x}, {north_y}).")
    
    # Step 4: Compute a Buffered Rectangular Bounding Box Around the Catchment
    minx, miny, maxx, maxy = selected_catchment.geometry.iloc[0].bounds
    minx -= buffer_distance
    miny -= buffer_distance
    maxx += buffer_distance
    maxy += buffer_distance
    
    # Step 5: Clip the Raster Using the Buffered Bounding Box
    with rasterio.open(raster_file) as src:
        # Convert bounding box coordinates to raster indices
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        
        # Read the subset of the raster
        clipped_data = src.read(1, window=window)
        
        # Compute new transform for the cropped raster
        new_transform = src.window_transform(window)
        
        # Step 6: Update metadata for output raster
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "transform": new_transform,
            "height": clipped_data.shape[0],
            "width": clipped_data.shape[1]
        })
        
        # Step 7: Save the clipped raster with buffer
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(clipped_data, 1)
    
    print(f" Clipped raster with {buffer_distance}m buffer saved to {output_raster}")

#########################################################################################################################

import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.mask import mask

def clip_raster_by_polygon(polygon_shapefile, raster_file, output_raster, polygon_id=None, id_field="ID", buffer_distance=0):
    """
    Clips a raster using the geometry of a polygon in a shapefile.

    Parameters:
        polygon_shapefile (str): Path to the shapefile containing clipping polygons.
        raster_file (str): Path to the input raster file.
        output_raster (str): Path to save the clipped raster.
        polygon_id (int, optional): ID of the polygon to use (from the shapefile). If None, uses all polygons.
        id_field (str): Name of the field in the shapefile that holds IDs.
        buffer_distance (float): Optional buffer distance in meters to expand the polygon.
    """
    
    # Load the polygon shapefile
    polygons = gpd.read_file(polygon_shapefile)

    # Filter by polygon ID if provided
    if polygon_id is not None:
        if id_field not in polygons.columns:
            raise ValueError(f"'{id_field}' field not found in shapefile.")
        selected = polygons[polygons[id_field] == polygon_id]
        if selected.empty:
            raise ValueError(f"No polygon found with {id_field} = {polygon_id}.")
    else:
        selected = polygons

    # Ensure the geometry is valid
    selected = selected[selected.is_valid]
    
    # Apply buffer if needed
    if buffer_distance != 0:
        selected = selected.copy()
        selected["geometry"] = selected.buffer(buffer_distance)

    # Clip the raster
    with rasterio.open(raster_file) as src:
        out_image, out_transform = mask(src, selected.geometry, crop=True)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f" Raster clipped to polygon{' with ID ' + str(polygon_id) if polygon_id is not None else ''} and saved to: {output_raster}")

############################################################################################################################## 
# Re-import necessary libraries after kernel reset
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point, box


def clip_dem_square_from_campground(
    campground_shapefile,
    csv_file,
    raster_file,
    output_raster,
    campground_id,
    square_extent=2048  # total size of square in meters (not half)
):
    """
    Clips a DEM raster in a square around a point inside a campground polygon,
    ensuring fixed extent for CUDA-safe resampling (e.g., 2048x2048m for 1024x1024 @ 2m).

    Parameters:
        campground_shapefile (str): Path to campground polygon shapefile.
        csv_file (str): CSV with 'ID', 'East_X', 'North_Y' coordinates.
        raster_file (str): Path to input DEM raster.
        output_raster (str): Output raster path.
        campground_id (int): ID of the campground.
        square_extent (float): Total square size in meters (default: 2048).
    """
    df = pd.read_csv(csv_file)
    if not {"ID", "East_X", "North_Y"}.issubset(df.columns):
        raise ValueError("CSV must contain 'ID', 'East_X', 'North_Y'.")

    row = df[df["ID"] == campground_id]
    if row.empty:
        raise ValueError(f"No point found for ID {campground_id}")

    x, y = row.iloc[0][["East_X", "North_Y"]]

    gdf = gpd.read_file(campground_shapefile)
    point = gpd.GeoDataFrame(geometry=[Point(x, y)], crs="EPSG:2056")
    point = point.to_crs(gdf.crs)

    selected_poly = gdf[gdf.contains(point.iloc[0].geometry)]
    if selected_poly.empty:
        raise ValueError("No campground polygon contains the point.")

    # Round bounds to fixed CUDA-safe square (e.g., 2048 x 2048 m)
    half_size = square_extent / 2
    center_x, center_y = x, y

    minx = round(center_x - half_size)
    maxx = round(center_x + half_size)
    miny = round(center_y - half_size)
    maxy = round(center_y + half_size)
    square_bbox = box(minx, miny, maxx, maxy)

    #  Check if polygon is fully inside the square
    campground_geom = selected_poly.geometry.iloc[0]
    if not campground_geom.within(square_bbox):
        print(" WARNING: The campground polygon is NOT fully within the clipped DEM bounds.")
        print(" Consider increasing 'square_extent' to ensure full coverage.")
    else:
        print(" The campground polygon is fully contained in the clipped DEM area.")

    #  Clip raster
    with rasterio.open(raster_file) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = src.window_transform(window)
        data = src.read(1, window=window)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform
        })

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(data, 1)

    print(f" Clipped DEM to fixed extent {square_extent} m saved to: {output_raster}")
    return output_raster

#########################################################################################################################
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point, box

def clip_dem_square_from_point(
    csv_file,
    raster_file,
    output_raster,
    campground_id,
    square_extent=2048  # Total square size in meters
):
    """
    Clips a DEM raster in a square around a point based on a given ID.

    Parameters:
        csv_file (str): CSV with 'ID', 'East_X', 'North_Y' coordinates.
        raster_file (str): Path to input DEM raster.
        output_raster (str): Output raster path.
        campground_id (int): ID of the point to center the square.
        square_extent (float): Total square size in meters (default: 2048).
    """
    df = pd.read_csv(csv_file)
    if not {"ID", "East_X", "North_Y"}.issubset(df.columns):
        raise ValueError("CSV must contain 'ID', 'East_X', 'North_Y'.")

    row = df[df["ID"] == campground_id]
    if row.empty:
        raise ValueError(f"No point found for ID {campground_id}")
    
    x, y = row.iloc[0][["East_X", "North_Y"]]

    half_size = square_extent / 2
    minx = round(x - half_size)
    maxx = round(x + half_size)
    miny = round(y - half_size)
    maxy = round(y + half_size)

    square_bbox = box(minx, miny, maxx, maxy)

    print(f" Clipping DEM around point ({x}, {y}) with extent {square_extent} m")

    #  Clip raster
    with rasterio.open(raster_file) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = src.window_transform(window)
        data = src.read(1, window=window)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform
        })

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(data, 1)

    print(f" Clipped DEM saved to: {output_raster}")
    return output_raster

###################################################################################################################
######### For Liestal and the case that we need a DEM divisible by 2 for the resampling##########################

import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
from rasterio.crs import CRS

def snap_to_grid(val, step=2, direction="down"):
    if direction == "down":
        return step * (val // step)
    else:
        return step * ((val + step - 1) // step)

def clip_dem_square_2m(
    csv_file,
    raster_file,
    output_raster,
    campground_id,
    square_extent=2048  # Should be divisible by 2!
):
    """
    Clips a DEM raster to a square extent around a selected point,
    snapping coordinates to a 2 m grid for LISFLOOD alignment.
    Ensures CRS is preserved or set to EPSG:2056 if missing.
    """
    df = pd.read_csv(csv_file)
    if not {"ID", "East_X", "North_Y"}.issubset(df.columns):
        raise ValueError("CSV must contain 'ID', 'East_X', 'North_Y'.")

    row = df[df["ID"] == campground_id]
    if row.empty:
        raise ValueError(f"No point found for ID {campground_id}")
    
    x, y = row.iloc[0][["East_X", "North_Y"]]

    half = square_extent / 2

    # Snap bounds to 2 m grid for alignment
    minx = snap_to_grid(x - half, step=2, direction="down")
    maxx = snap_to_grid(x + half, step=2, direction="up")
    miny = snap_to_grid(y - half, step=2, direction="down")
    maxy = snap_to_grid(y + half, step=2, direction="up")

    print(f" Snapped extent: X: {minx} → {maxx}, Y: {miny} → {maxy}")
    print(f" Final grid size: {int((maxy - miny) / 2)} rows × {int((maxx - minx) / 2)} cols")

    # Clip raster
    with rasterio.open(raster_file) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = src.window_transform(window)
        data = src.read(1, window=window)

        # Assign CRS if missing
        crs = src.crs or CRS.from_epsg(2056)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform,
            "crs": crs
        })

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(data, 1)

    print(f" Clipped DEM saved to: {output_raster}")
    return output_raster


###########################################################################################################################

import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.windows import from_bounds

def clip_raster_by_polygon_bbox(polygon_shapefile, raster_file, output_raster, polygon_id=None, id_field="ID", buffer_distance=0):
    """
    Clips a raster using the square bounding box around a polygon from a shapefile.

    Parameters:
        polygon_shapefile (str): Path to shapefile containing polygons.
        raster_file (str): Path to input raster.
        output_raster (str): Path to save clipped raster.
        polygon_id (int, optional): ID of the polygon to use.
        id_field (str): Field name in the shapefile for ID matching.
        buffer_distance (float): Buffer in meters around the polygon.
    """
    # Load polygon shapefile
    polygons = gpd.read_file(polygon_shapefile)

    # Filter by ID if needed
    if polygon_id is not None:
        if id_field not in polygons.columns:
            raise ValueError(f"'{id_field}' not found in shapefile.")
        selected = polygons[polygons[id_field] == polygon_id]
        if selected.empty:
            raise ValueError(f"No polygon found with {id_field} = {polygon_id}")
    else:
        selected = polygons

    # Ensure valid geometry
    selected = selected[selected.is_valid]
    if selected.empty:
        raise ValueError("No valid geometries found.")

    # Get bounding box of the unioned geometry
    bounds = selected.unary_union.buffer(buffer_distance).bounds
    minx, miny, maxx, maxy = bounds

    # Make bounding box square (quadratic)
    x_diff = maxx - minx
    y_diff = maxy - miny
    if x_diff > y_diff:
        padding = (x_diff - y_diff) / 2
        miny -= padding
        maxy += padding
    else:
        padding = (y_diff - x_diff) / 2
        minx -= padding
        maxx += padding

    # Clip raster by window (square bounding box)
    with rasterio.open(raster_file) as src:
        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        transform = src.window_transform(window)
        data = src.read(1, window=window)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform
        })

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(data, 1)

    print(f"Raster clipped to square bounding box of polygon{' with ID ' + str(polygon_id) if polygon_id else ''} and saved to: {output_raster}")

##########################################################################################################################
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

def resample_raster(input_raster, output_raster, target_resolution, resampling_method):
    """
    Resamples a raster to the desired resolution using the specified resampling method,
    preserving the CRS and metadata.
    
    Parameters:
        input_raster (str): Path to the input raster file.
        output_raster (str): Path to save the resampled raster.
        target_resolution (float): Desired output resolution in meters.
        resampling_method (str): Resampling method ('nearest', 'cubic', or 'bilinear').
    """
    resampling_methods = {
        "nearest": Resampling.nearest,
        "cubic": Resampling.cubic,
        "bilinear": Resampling.bilinear
    }
    
    if resampling_method not in resampling_methods:
        raise ValueError("Invalid resampling method. Choose 'nearest', 'cubic', or 'bilinear'.")

    with rasterio.open(input_raster) as src:
        new_width = int(src.width * (src.res[0] / target_resolution))
        new_height = int(src.height * (src.res[1] / target_resolution))

        transform = rasterio.transform.from_bounds(*src.bounds, new_width, new_height)

        with WarpedVRT(src, transform=transform, width=new_width, height=new_height,
                       resampling=resampling_methods[resampling_method]) as vrt:
            resampled_data = vrt.read(1)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "transform": transform,
            "height": new_height,
            "width": new_width,
            "crs": src.crs  #  Ensure CRS is preserved
        })

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(resampled_data, 1)

    print(f" Resampled raster ({target_resolution} m resolution, {resampling_method} method) saved to {output_raster}")

#####################################################################################################
############## RESAMPLING THE RASTER OPTION 2 - THAT gives a 


######################################################################################################
####### CLIP BASED ON THE AREALSTATISTIK FILE QUADRATIC THE DEM FILE IN THE SAME SIZE################
#####################################################################################################

import rasterio
from rasterio.windows import from_bounds

def clip_dem_to_match_areal(clip_raster_path, input_raster_path, output_clipped_raster_path):
    """
    Clips a raster to match the bounding box of another raster.
    
    Parameters:
        clip_raster_path (str): Path to the raster used for clipping (e.g., Arealstatistik raster).
        input_raster_path (str): Path to the raster that needs to be clipped (e.g., DEM raster).
        output_clipped_raster_path (str): Path to save the clipped raster.
    """
    # Step 1: Get Bounding Box from Clipping Raster
    with rasterio.open(clip_raster_path) as clip_src:
        minx, miny, maxx, maxy = clip_src.bounds  # Extract bounding box

    # Step 2: Use the Bounding Box to Clip the Target Raster
    with rasterio.open(input_raster_path) as src:
        # Convert bounding box coordinates to raster indices
        window = from_bounds(minx, miny, maxx, maxy, src.transform)

        # Read the subset of the raster
        clipped_data = src.read(1, window=window)

        # Compute new transform for the cropped raster
        new_transform = src.window_transform(window)

        # Step 3: Update metadata for output raster
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "transform": new_transform,
            "height": clipped_data.shape[0],
            "width": clipped_data.shape[1]
        })

        # Step 4: Save the clipped raster
        with rasterio.open(output_clipped_raster_path, "w", **out_meta) as dest:
            dest.write(clipped_data, 1)

    print(f" Clipped raster saved to {output_clipped_raster_path}")


########################################################################################################

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



################################################################################################
##########CREATE A SINGLE NETCDFILE FOR A VALUE #############################################
################################################################################################

import numpy as np
import rasterio
from netCDF4 import Dataset

def precipitation_netcdf(dem_file, total_precipitation, nc_path, frequency_per_hour):
    """
    Function to create a NetCDF file for precipitation based on the total precipitation and a predefined frequency distribution.

    Parameters:
    - dem_file (str): Path to the DEM raster file (GeoTIFF).
    - total_precipitation (float): The total precipitation value (e.g., 48 mm).
    - nc_path (str): Path where the output NetCDF file will be saved.
    - frequency_per_hour (list): List of precipitation distribution based on frequency per hour (mm per hour).

    Returns:
    - None
    """

    # Step 1: Open the DEM raster to retrieve metadata and its spatial information
    with rasterio.open(dem_file) as src:
        width = src.width
        height = src.height
        transform = src.transform  # Affine transformation
        x_min, y_max = src.bounds.left, src.bounds.top
        cellsize = src.res[0]  # Assuming square cells
        x_coords = np.arange(x_min, x_min + cellsize * width, cellsize)
        y_coords = np.arange(y_max, y_max - cellsize * height, -cellsize)

    # Step 2: Calculate the total frequency and scale the precipitation
    total_frequency = sum(frequency_per_hour)  # Sum of all the frequency values (total distribution)
    
    # Calculate the precipitation values for each time step (proportional to frequency)
    precipitation_values = np.array([total_precipitation * (f / total_frequency) for f in frequency_per_hour])
    
    # Step 3: Create a 3D array for precipitation (time, y, x)
    time_steps = len(frequency_per_hour)
    precipitation_data = np.zeros((time_steps, height, width), dtype=np.float32)
    
    # Fill precipitation data with the calculated values
    for t, value in enumerate(precipitation_values):
        precipitation_data[t, :, :] = value

    # Step 4: Convert time to fractional hours (assuming 5-minute intervals)
    time_in_hours = np.arange(0, time_steps) * (5 / 60)  # Convert 5-minute intervals to fractional hours

    # Step 5: Create the NetCDF file
    with Dataset(nc_path, "w", format="NETCDF4") as nc:
        # Create dimensions
        nc.createDimension("time", time_steps)
        nc.createDimension("y", height)
        nc.createDimension("x", width)
        
        # Create coordinate variables
        times = nc.createVariable("time", "f4", ("time",))
        ys = nc.createVariable("y", "f4", ("y",))
        xs = nc.createVariable("x", "f4", ("x",))
        
        # Assign coordinate data
        times[:] = time_in_hours  # Fractional hours
        ys[:] = y_coords
        xs[:] = x_coords
        
        # Assign attributes for coordinate variables
        times.long_name = "time"
        ys.long_name = "y-coordinate in projected units"
        xs.long_name = "x-coordinate in projected units"
        
        # Create the precipitation variable
        precip = nc.createVariable("rainfall_depth", "f4", ("time", "y", "x"), zlib=True)
        precip.units = "mm"
        precip.long_name = "Rainfall depth"
        precip.standard_name = "precipitation_amount"
        precip[:] = precipitation_data  # Assign precipitation data
        
        # Add global attributes
        nc.description = "Spatially and temporally varying rainfall for TUFLOW model, with fractional hours for time"
        nc.history = "Created on: 2025-01-25"
        nc.source = "Generated from Zell DEM and 30-year return period rainfall data"

    print(f"TUFLOW-compatible NetCDF file created: {nc_path}")

    
########################################################################################################################
################################################################################################
##########CREATE A SINGLE NETCDFILE FOR DIFFERENT SCENARIOS #############################################
################################################################################################

import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.warp import reproject, Resampling

def precipitation_variosnetcdfiles(dem_file, total_precipitation_values, output_folder, distribution_percentages, resolution=100):
    """
    Function to generate multiple precipitation scenarios for different total precipitation values.
    
    Parameters:
    - dem_file (str): Path to the input DEM file (GeoTIFF) for spatial information.
    - total_precipitation_values (list): List of total precipitation values (e.g., [48, 100, 150]) to generate scenarios.
    - output_folder (str): Folder where the NetCDF files will be saved.
    - distribution_percentages (list): The distribution percentages (e.g., based on the 5-minute intervals).
    - resolution (int): The desired resolution of the raster in map units (default is 100m).
    """
    
    # Open the DEM raster to retrieve metadata and its mask
    with rasterio.open(dem_file) as src:
        width = src.width
        height = src.height
        transform = src.transform  # Affine transformation
        x_min, y_max = src.bounds.left, src.bounds.top
        cellsize = src.res[0]  # Assuming square cells
        x_coords = np.arange(x_min, x_min + cellsize * width, cellsize)
        y_coords = np.arange(y_max, y_max - cellsize * height, -cellsize)
    
    # Number of time steps (12 for every 5 minutes up to 60 minutes)
    time_steps = len(distribution_percentages)
    
    # Iterate over the total precipitation values to generate different scenarios
    for total_precipitation in total_precipitation_values:
        # Calculate the precipitation values for each time step based on the distribution
        precipitation_values = [total_precipitation * (percent / 100) for percent in distribution_percentages]
        
        # Create a 3D array for precipitation (time, y, x)
        precipitation_data = np.zeros((time_steps, height, width), dtype=np.float32)
        
        # Fill precipitation data with calculated values
        for t, value in enumerate(precipitation_values):
            precipitation_data[t, :, :] = value
        
        # Convert time to fractional hours
        time_in_hours = np.arange(0, time_steps) * (5 / 60)  # Convert 5-minute intervals to fractional hours
        
        # Prepare the output NetCDF filename
        output_nc_file = f"{output_folder}/precipitation_{total_precipitation}_mm.nc"
        
        # Create the NetCDF file
        with Dataset(output_nc_file, "w", format="NETCDF4") as nc:
            # Create dimensions
            nc.createDimension("time", time_steps)
            nc.createDimension("y", height)
            nc.createDimension("x", width)
            
            # Create coordinate variables
            times = nc.createVariable("time", "f4", ("time",))
            ys = nc.createVariable("y", "f4", ("y",))
            xs = nc.createVariable("x", "f4", ("x",))
            
            # Assign coordinate data
            times[:] = time_in_hours  # Fractional hours
            ys[:] = y_coords
            xs[:] = x_coords
            
            # Assign attributes for coordinate variables
            times.long_name = "time"
            ys.long_name = "y-coordinate in projected units"
            xs.long_name = "x-coordinate in projected units"
            
            # Create the precipitation variable
            precip = nc.createVariable("rainfall_depth", "f4", ("time", "y", "x"), zlib=True)
            precip.units = "mm"
            precip.long_name = "Rainfall depth"
            precip.standard_name = "precipitation_amount"
            precip[:] = precipitation_data  # Assign precipitation data
            
            # Add global attributes
            nc.description = "Spatially and temporally varying rainfall for TUFLOW model, with fractional hours for time"
            nc.history = "Created on: 2025-03-08"
            nc.source = f"Generated from Zell DEM and different total precipitation values, scenario {total_precipitation} mm"
        
        print(f"TUFLOW-compatible NetCDF file created: {output_nc_file}")

#################################################################################################################################
###### extension change ################

import os
import rasterio
from rasterio import shutil as rio_shutil

def convert_and_rename_raster(input_file_path, output_extension):
    """
    Converts a raster to the specified format while preserving CRS and metadata.

    Parameters:
        input_file_path (str): Path to the input raster (e.g., .tif).
        output_extension (str): New extension, e.g., '.asc' or '.n'.

    Returns:
        str: Path to the converted file.
    """
    base = os.path.splitext(input_file_path)[0]
    new_file_path = f"{base}{output_extension}"

    if not os.path.exists(input_file_path):
        print(f" File does not exist: {input_file_path}")
        return None

    with rasterio.open(input_file_path) as src:
        profile = src.profile.copy()

        # Handle format based on extension
        if output_extension == ".asc":
            profile.update({
                "driver": "AAIGrid",  # ASCII Grid
                "dtype": "float32",
                "crs": src.crs
            })
        elif output_extension == ".n":
            profile.update({
                "driver": "AAIGrid",  # LISFLOOD may need AAIGrid format renamed
                "dtype": "float32",
                "crs": src.crs
            })
        else:
            raise ValueError(f"Unsupported extension for conversion: {output_extension}")

        with rasterio.open(new_file_path, "w", **profile) as dst:
            dst.write(src.read(1), 1)

    print(f" Raster converted and saved as: {new_file_path}")
    return new_file_path

#######################################################################

def create_par_file(base_name, total_precipitation, output_file_path):
    """
    Creates a .par text file with the specified data and appends the total precipitation value 
    to the 'resroot' field and creates other parameters based on the base name.

    Parameters:
    - base_name (str): The base name of the file, e.g., 'Salavaux_catchment_perc999'.
    - total_precipitation (str): The total precipitation value (e.g., '48' for '_48' in 'resroot').
    - output_file_path (str): Path to save the output .par file.
    """
    
    # Define the file paths using the base name and extension
    file_data = {
        "DEMfile": f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile": f"{base_name}.stage",
        "dirroot": base_name,  # Using the base name as dirroot
        "manningfile": f"{base_name}.n",
        "resroot": f"{base_name}_1h_{total_precipitation}",  # Append total_precipitation to resroot
        "sim_time": "3600.0",
        "initial_tstep": "1.0",
        "saveint": "300.0",
        "massint": "1.0",
        "netcdf_out": ""
    }
    
    # Create the .par file
    try:
        with open(output_file_path, 'w') as f:
            # Write a header to separate parameter names and values
            f.write("# Parameters and Values\n\n")
            
            # Write each key-value pair in the dictionary to the .par file in two-column format
            for key, value in file_data.items():
                f.write(f"{key:25} {value}\n")
                
            print(f"File created successfully at {output_file_path}")
    except Exception as e:
        print(f"Error occurred: {e}")

##########################################################################

def create_par_file_Liestal_5hour(base_name,ensemble, output_file_path):
    """
    Creates a .par text file with the specified data and appends the total precipitation value 
    to the 'resroot' field and creates other parameters based on the base name.

    Parameters:
    - base_name (str): The base name of the file, e.g., 'Salavaux_catchment_perc999'.
    - total_precipitation (str): The total precipitation value (e.g., '48' for '_48' in 'resroot').
    - output_file_path (str): Path to save the output .par file.
    """
    
    # Define the file paths using the base name and extension
    file_data = {
        "DEMfile": f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile": f"{base_name}.stage",
        "dirroot": base_name,  # Using the base name as dirroot
        "manningfile": f"{base_name}.n",
        "resroot": f"{base_name}_{ensemble}",  # Append total_precipitation to resroot
        "sim_time": "18000.0",
        "initial_tstep": "1.0",
        "saveint": "300.0",
        "massint": "1.0",
        "netcdf_out": ""
    }
    
    # Create the .par file
    try:
        with open(output_file_path, 'w') as f:
            # Write a header to separate parameter names and values
            f.write("# Parameters and Values\n\n")
            
            # Write each key-value pair in the dictionary to the .par file in two-column format
            for key, value in file_data.items():
                f.write(f"{key:25} {value}\n")
                
            print(f"File created successfully at {output_file_path}")
    except Exception as e:
        print(f"Error occurred: {e}")

############################################################################

def create_par_file_Liestal_Combiprecip(base_name, output_file_path):
    """
    Creates a .par text file for a single deterministic run (no ensemble index).
    
    Parameters:
    - base_name (str): Base name of the files, e.g., 'Liestal_2m'.
    - output_file_path (str): Full path to save the .par file.
    """

    # Define file references
    file_data = {
        "DEMfile": f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile": f"{base_name}.stage",
        "dirroot": base_name,
        "manningfile": f"{base_name}.n",
        "resroot": base_name,  # No _1 or ensemble suffix
        "sim_time": "36000.0",
        "initial_tstep": "1.0",
        "saveint": "300.0",
        "massint": "1.0",
        "fv1": "",
        "cuda": "",
        "netcdf_out": ""
    }

    # Write the .par file
    try:
        with open(output_file_path, 'w') as f:
            f.write("# Parameters and Values\n\n")
            for key, value in file_data.items():
                f.write(f"{key:25} {value}\n")
        print(f"✔ File created at: {output_file_path}")
    except Exception as e:
        print(f" Error: {e}")
##############################################################################

def create_par_file_Combiprecip(base_name, time, output_file_path):
    """
    Creates a .par text file for a single deterministic run (no ensemble index).
    'time' should be given in seconds.
    """
    file_data = {
        "DEMfile": f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile": f"{base_name}.stage",
        "dirroot": base_name,
        "manningfile": f"{base_name}.n",
        "bcifile":f"{base_name}.bci",
        "resroot": base_name,  # No _1 or ensemble suffix
        "sim_time": str(time),  # <-- fixed!
        "initial_tstep": "1.0",
        "saveint": "3600.0",
        "massint": "1.0",
        "fv1": "",
        "cuda": "",
        "netcdf_out": ""
    }

    try:
        with open(output_file_path, 'w') as f:
            f.write("# Parameters and Values\n\n")
            for key, value in file_data.items():
                f.write(f"{key:25} {value}\n")
        print(f"✔ File created at: {output_file_path}")
    except Exception as e:
        print(f" Error: {e}")
############################################################################

def create_par_file_Combiprecip_bdy(base_name, time, output_file_path):
    """
    Creates a .par text file for a single deterministic run (no ensemble index).
    'time' should be given in seconds.
    """
    file_data = {
        "DEMfile": f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile": f"{base_name}.stage",
        "dirroot": base_name,
        "manningfile": f"{base_name}.n",
        "bcifile":f"{base_name}.bci",
        "bdyfile":f"{base_name}.bdy",
        "resroot": base_name,  # No _1 or ensemble suffix
        "sim_time": str(time),  # <-- fixed!
        "initial_tstep": "1.0",
        "saveint": "3600.0",
        "massint": "1.0",
        "fv1": "",
        "cuda": "",
        "netcdf_out": ""
    }

    try:
        with open(output_file_path, 'w') as f:
            f.write("# Parameters and Values\n\n")
            for key, value in file_data.items():
                f.write(f"{key:25} {value}\n")
        print(f"✔ File created at: {output_file_path}")
    except Exception as e:
        print(f" Error: {e}")

#############################################################################
def create_par_file_Combiprecip_bdy_qoutput(base_name, time, cfl, output_file_path):
    """
    Creates a .par text file for a single deterministic run (no ensemble index).
    'time' should be given in seconds.
    """
    file_data = {
        "DEMfile":         f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile":       f"{base_name}.stage",
        "dirroot":         base_name,
        "manningfile":     f"{base_name}.n",
        "bcifile":         f"{base_name}.bci",
        "bdyfile":         f"{base_name}.bdy",
        "resroot":         base_name,   # No _1 or ensemble suffix
        "sim_time":        str(time),
        "initial_tstep":   "1.0",
        "saveint":         "3600.0",
        "massint":         "1.0",
        "cfl":             str(cfl),    # <-- now uses the parameter
        # flags: write as bare keys (no value)
        "fv1":             "",
        "cuda":            "",
        "qoutput":         "",
        "voutput":         "",
        "netcdf_out":      "",
    }

    try:
        with open(output_file_path, 'w') as f:
            f.write("# Parameters and Values\n\n")
            for key, value in file_data.items():
                if value == "":
                    f.write(f"{key}\n")               # bare flag line
                else:
                    f.write(f"{key:25} {value}\n")    # key/value line
        print(f"✔ File created at: {output_file_path}")
    except Exception as e:
        print(f" Error: {e}")
        
############################################################################

#############################################################################
def create_par_file_Forecast_bdy_qoutput(base_name, time, cfl, output_file_path):
    """
    Creates a .par text file for a single deterministic run (no ensemble index).
    'time' should be given in seconds.
    """
    file_data = {
        "DEMfile":         f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile":       f"{base_name}.stage",
        "dirroot":         base_name,
        "manningfile":     f"{base_name}.n",
        "bcifile":         f"{base_name}.bci",
        "bdyfile":         f"{base_name}.bdy",
        "resroot":         base_name,   # No _1 or ensemble suffix
        "sim_time":        str(time),
        "initial_tstep":   "1.0",
        "saveint":         "3600.0",
        "massint":         "1.0",
        "cfl":             str(cfl),    # <-- now uses the parameter
        # flags: write as bare keys (no value)
        #"fv1":             "",
        #"cuda":            "",
        "qoutput":         "",
        "voutput":         "",
        "netcdf_out":      "",
    }

    try:
        with open(output_file_path, 'w') as f:
            f.write("# Parameters and Values\n\n")
            for key, value in file_data.items():
                if value == "":
                    f.write(f"{key}\n")               # bare flag line
                else:
                    f.write(f"{key:25} {value}\n")    # key/value line
        print(f"✔ File created at: {output_file_path}")
    except Exception as e:
        print(f" Error: {e}")


#############################################################################

def create_par_file_Combiprecip_bdy_dg2(base_name, time, output_file_path):
    """
    Creates a .par text file for a single deterministic run (no ensemble index).
    'time' should be given in seconds.
    """
    file_data = {
        "DEMfile": f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile": f"{base_name}.stage",
        "dirroot": base_name,
        "manningfile": f"{base_name}.n",
        "bcifile":f"{base_name}.bci",
        "bdyfile":f"{base_name}.bdy",
        "resroot": base_name,  # No _1 or ensemble suffix
        "sim_time": str(time),  # <-- fixed!
        "initial_tstep": "1.0",
        "saveint": "3600.0",
        "massint": "1.0",
        "dg2": "",
        "limitslopes": "",
        "cuda": "",
        "netcdf_out": ""
    }

    try:
        with open(output_file_path, 'w') as f:
            f.write("# Parameters and Values\n\n")
            for key, value in file_data.items():
                f.write(f"{key:25} {value}\n")
        print(f"✔ File created at: {output_file_path}")
    except Exception as e:
        print(f" Error: {e}")

##############################################################################

def create_par_file_Liestal_10hour(base_name,ensemble, output_file_path):
    """
    Creates a .par text file with the specified data and appends the total precipitation value 
    to the 'resroot' field and creates other parameters based on the base name.

    Parameters:
    - base_name (str): The base name of the file, e.g., 'Salavaux_catchment_perc999'.
    - total_precipitation (str): The total precipitation value (e.g., '48' for '_48' in 'resroot').
    - output_file_path (str): Path to save the output .par file.
    """
    
    # Define the file paths using the base name and extension
    file_data = {
        "DEMfile": f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile": f"{base_name}.stage",
        "dirroot": base_name,  # Using the base name as dirroot
        "manningfile": f"{base_name}.n",
        "resroot": f"{base_name}_{ensemble}",  # Append total_precipitation to resroot
        "sim_time": "36000.0",
        "initial_tstep": "1.0",
        "saveint": "300.0",
        "massint": "1.0",
        "netcdf_out": ""
    }
    
    # Create the .par file
    try:
        with open(output_file_path, 'w') as f:
            # Write a header to separate parameter names and values
            f.write("# Parameters and Values\n\n")
            
            # Write each key-value pair in the dictionary to the .par file in two-column format
            for key, value in file_data.items():
                f.write(f"{key:25} {value}\n")
                
            print(f"File created successfully at {output_file_path}")
    except Exception as e:
        print(f"Error occurred: {e}")

###########################################################################

def create_par_file_Liestal_33hour(base_name,ensemble, output_file_path):
    """
    Creates a .par text file with the specified data and appends the total precipitation value 
    to the 'resroot' field and creates other parameters based on the base name.

    Parameters:
    - base_name (str): The base name of the file, e.g., 'Salavaux_catchment_perc999'.
    - total_precipitation (str): The total precipitation value (e.g., '48' for '_48' in 'resroot').
    - output_file_path (str): Path to save the output .par file.
    """
    
    # Define the file paths using the base name and extension
    file_data = {
        "DEMfile": f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile": f"{base_name}.stage",
        "dirroot": base_name,  # Using the base name as dirroot
        "manningfile": f"{base_name}.n",
        "resroot": f"{base_name}_{ensemble}",  # Append total_precipitation to resroot
        "sim_time": "118800.0",
        "initial_tstep": "1.0",
        "saveint": "300.0",
        "massint": "1.0",
        "netcdf_out": ""
    }
    
    # Create the .par file
    try:
        with open(output_file_path, 'w') as f:
            # Write a header to separate parameter names and values
            f.write("# Parameters and Values\n\n")
            
            # Write each key-value pair in the dictionary to the .par file in two-column format
            for key, value in file_data.items():
                f.write(f"{key:25} {value}\n")
                
            print(f"File created successfully at {output_file_path}")
    except Exception as e:
        print(f"Error occurred: {e}")

#########################################################################

def create_par_file_fv1(base_name, total_precipitation, output_file_path):
    """
    Creates a .par text file with the specified data and appends the total precipitation value 
    to the 'resroot' field and creates other parameters based on the base name.

    Parameters:
    - base_name (str): The base name of the file, e.g., 'Salavaux_catchment_perc999'.
    - total_precipitation (str): The total precipitation value (e.g., '48' for '_48' in 'resroot').
    - output_file_path (str): Path to save the output .par file.
    """
    
    # Define the file paths using the base name and extension
    file_data = {
        "DEMfile": f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile": f"{base_name}.stage",
        "dirroot": base_name,  # Using the base name as dirroot
        "manningfile": f"{base_name}.n",
        "resroot": f"{base_name}_1h_{total_precipitation}",  # Append total_precipitation to resroot
        "fv1": "",
        "cuda": "",
        "sim_time": "3600.0",
        "initial_tstep": "1.0",
        "saveint": "300.0",
        "massint": "300.0",
        "netcdf_out": ""
    }
    
    # Create the .par file
    try:
        with open(output_file_path, 'w') as f:
            # Write a header to separate parameter names and values
            f.write("# Parameters and Values\n\n")
            
            # Write each key-value pair in the dictionary to the .par file in two-column format
            for key, value in file_data.items():
                f.write(f"{key:25} {value}\n")
                
            print(f"File created successfully at {output_file_path}")
    except Exception as e:
        print(f"Error occurred: {e}")
#####################################################################################################

def create_par_file_acc(base_name, total_precipitation, output_file_path):
    """
    Creates a .par text file with the specified data and appends the total precipitation value 
    to the 'resroot' field and creates other parameters based on the base name.

    Parameters:
    - base_name (str): The base name of the file, e.g., 'Salavaux_catchment_perc999'.
    - total_precipitation (str): The total precipitation value (e.g., '48' for '_48' in 'resroot').
    - output_file_path (str): Path to save the output .par file.
    """
    
    # Define the file paths using the base name and extension
    file_data = {
        "DEMfile": f"{base_name}.dem",
        "dynamicrainfile": f"{base_name}.nc",
        "stagefile": f"{base_name}.stage",
        "dirroot": base_name,  # Using the base name as dirroot
        "manningfile": f"{base_name}.n",
        "resroot": f"{base_name}_1h_{total_precipitation}",  # Append total_precipitation to resroot
        "acceleration": "",
        "sim_time": "3600.0",
        "initial_tstep": "1.0",
        "saveint": "300.0",
        "massint": "1.0",
        "netcdf_out": ""
    }
    
    # Create the .par file
    try:
        with open(output_file_path, 'w') as f:
            # Write a header to separate parameter names and values
            f.write("# Parameters and Values\n\n")
            
            # Write each key-value pair in the dictionary to the .par file in two-column format
            for key, value in file_data.items():
                f.write(f"{key:25} {value}\n")
                
            print(f"File created successfully at {output_file_path}")
    except Exception as e:
        print(f"Error occurred: {e}")

###################################################################################################

import pandas as pd

def create_stage_file(catchment_location_csv, selected_id, output_stage_file, num_points=1):
    """
    Function to create a .stage file based on selected ID from the catchment location CSV.
    
    This version includes the number of points (num_points) in the first row and coordinates in the second row.
    The columns are separated by tabs for better readability.

    Parameters:
    - catchment_location_csv (str): Path to the CSV file containing catchment location data.
    - selected_id (int): The ID for the catchment location to select (e.g., '1').
    - output_stage_file (str): The path to save the output .stage file.
    - num_points (int): The number of points to write in the first row (default is 1).
    """
    # Read the CSV file into a DataFrame
    df_location = pd.read_csv(catchment_location_csv)
    
    # Select the row corresponding to the provided ID
    selected_row = df_location[df_location['ID'] == selected_id]
    
    if selected_row.empty:
        raise ValueError(f"ID {selected_id} not found in the CSV file.")
    
    # Extract the East_X and North_Y values
    east_x = selected_row['East_X'].values[0]
    north_y = selected_row['North_Y'].values[0]
    
    # Create the .stage file content
    stage_data = f"{num_points}\n{east_x}\t{north_y}\n"
    
    # Write the data to the .stage file
    try:
        with open(output_stage_file, 'w') as f:
            f.write(stage_data)
        print(f".stage file created successfully at {output_stage_file}")
    except Exception as e:
        print(f"Error occurred: {e}")

#################################################################################
#######################################################################################################
###################################################################################################
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