from typing import Dict
from typing import Dict, List, Tuple
import os, re, glob
import numpy as np
import xarray as xr
import rioxarray as rxr

########## PLOTTING ############################################ 

import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd

def g_plots_from_wd_swissimage_files(dem_file, wd_folder, plot_output_folder, geo_ezgg_2km_ge, swissimage_file, plot_title_prefix):
    """
    Generates plots of water depth data over a Swissimage background using a DEM grid.

    Parameters:
        dem_file (str): Path to the .dem file.
        wd_folder (str): Path to the folder containing .wd files.
        plot_output_folder (str): Path to save the plots.
        geo_ezgg_2km_ge (str): Path to the catchment shapefile.
        swissimage_file (str): Path to the Swissimage background (.tif).
        plot_title_prefix (str): Prefix for plot titles.
    """
    # Ensure the output directory exists
    os.makedirs(plot_output_folder, exist_ok=True)

    # Step 1: Read the DEM grid structure and mask
    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        mask = dem_data != dem_nodata_value

    # Step 2: Read the Swissimage background in RGB
    with rasterio.open(swissimage_file) as src_swissimage:
        swissimage_data = src_swissimage.read([1, 2, 3])
        swissimage_bounds = src_swissimage.bounds

    # Step 3: Read the catchment shapefile
    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")

    # Step 4: Iterate through all .wd files in the folder
    wd_files = sorted([os.path.join(wd_folder, f) for f in os.listdir(wd_folder) if f.endswith(".wd")])

    for i, wd_file in enumerate(wd_files):
        try:
            # Read water depth data
            with rasterio.open(wd_file) as src_wd:
                wd_data = src_wd.read(1)
                wd_transform = src_wd.transform

            # Reproject water depth data to match DEM grid
            aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=wd_data,
                destination=aligned_data,
                src_transform=wd_transform,
                src_crs="EPSG:2056",
                dst_transform=dem_transform,
                dst_crs="EPSG:2056",
                resampling=Resampling.nearest,
            )

            # Mask and categorize data
            masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
            transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

            categories = [0.10, 0.25, 0.50, 0.60]
            colors = ['#ffffcc', '#ffeda0', '#0047b3']
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(categories, cmap.N, clip=True)

            # Plot the data
            plt.figure(figsize=(12, 10))
            plt.imshow(
                np.moveaxis(swissimage_data, 0, -1),
                extent=(swissimage_bounds.left, swissimage_bounds.right, swissimage_bounds.bottom, swissimage_bounds.top),
                interpolation="none",
                zorder=0,
                alpha=0.9,
            )

            # Overlay transparent and masked data
            plt.imshow(transparent_data, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top), cmap=ListedColormap(['none']), interpolation="none", zorder=1)
            plt.imshow(masked_data, cmap=cmap, norm=norm, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top), interpolation="none", zorder=2)

            # Overlay catchment boundaries
            catchments.boundary.plot(ax=plt.gca(), edgecolor="black", linewidth=0.7, zorder=3, label="Catchment Boundary")

            # Customize plot
            plt.xlim(dem_bounds.left, dem_bounds.right)
            plt.ylim(dem_bounds.bottom, dem_bounds.top)

            # Colorbar
            cbar = plt.colorbar(label="Water Depth (m)", boundaries=categories, ticks=[0.10, 0.25, 0.50])
            cbar.set_ticklabels(["0.10m", "0.25m", "> 0.50 m"])

            # Title and labels
            time_minutes = i * 5
            plot_title = f"{plot_title_prefix} - {time_minutes} minutes"
            plt.title(plot_title)
            plt.xlabel("Longitude (m)")
            plt.ylabel("Latitude (m)")
            plt.legend(loc="upper right")

            # Save plot
            plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(os.path.basename(wd_file))[0]}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Plot saved: {plot_filename}")

        except Exception as e:
            print(f"Failed to process {wd_file}: {e}")

    print("All plots have been generated and saved.")


    ##############################################################

import os
import re
from PIL import Image

def create_gif_from_images(image_folder, output_gif, duration=500, start=0, end=12):
    """
    Creates a GIF from PNG images in a specified folder.

    Parameters:
        image_folder (str): Path to the folder containing PNG files.
        output_gif (str): Path to save the GIF file.
        duration (int): Duration between frames in milliseconds. Default is 500.
        start (int): Starting index of images to include in the GIF. Default is 0.
        end (int): Ending index of images to include in the GIF. Default is 12.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    # Get all PNG files matching the expected naming pattern, e.g., xxx-0001.png
    image_files = sorted(
        [f for f in os.listdir(image_folder) if re.match(r".*-\d{4}\.png$", f)],
        key=lambda x: int(x.split('-')[-1].split('.')[0])
    )

    # Filter files from the specified range
    filtered_files = [f for f in image_files if start <= int(f.split('-')[-1].split('.')[0]) <= end]

    if len(filtered_files) == 0:
        print(f"No images found in the specified range ({start:04d} to {end:04d}).")
    else:
        print(f"Found {len(filtered_files)} images. Creating GIF...")

        # Load images
        images = [Image.open(os.path.join(image_folder, f)) for f in filtered_files]

        # Save as GIF
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0  # Infinite loop
        )

        print(f" GIF created successfully: {output_gif}")


####################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd

def generate_plot_for_max_file(
    dem_file, 
    max_file, 
    plot_output_folder, 
    geo_ezgg_2km_ge, 
    swissimage_file, 
    plot_title
):
    """
    Generates a plot of water depth from a `.max` file over a Swissimage background using a DEM grid.

    Parameters:
        dem_file (str): Path to the .dem file.
        max_file (str): Path to the .max file.
        plot_output_folder (str): Path to save the plot.
        geo_ezgg_2km_ge (str): Path to the catchment shapefile.
        swissimage_file (str): Path to the Swissimage background (.tif).
        plot_title (str): Title for the plot.
    """
    # Ensure the output directory exists
    os.makedirs(plot_output_folder, exist_ok=True)

    # Step 1: Read the DEM grid structure and mask
    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        mask = dem_data != dem_nodata_value

    # Step 2: Read the Swissimage background in RGB
    with rasterio.open(swissimage_file) as src_swissimage:
        swissimage_data = src_swissimage.read([1, 2, 3])
        swissimage_bounds = src_swissimage.bounds

    # Step 3: Read the catchment shapefile
    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")

    try:
        # Read water depth data from .max file
        with rasterio.open(max_file) as src_max:
            max_data = src_max.read(1)
            max_transform = src_max.transform

        # Reproject water depth data to match DEM grid
        aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
        reproject(
            source=max_data,
            destination=aligned_data,
            src_transform=max_transform,
            src_crs="EPSG:2056",
            dst_transform=dem_transform,
            dst_crs="EPSG:2056",
            resampling=Resampling.nearest,
        )

        # Mask and categorize data
        masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
        transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

        categories = [0.10, 0.25, 0.50, 0.60]
        colors = ['#ffffcc', '#ffeda0', '#0047b3']
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(categories, cmap.N, clip=True)

        # Plot the data
        plt.figure(figsize=(12, 10))
        plt.imshow(
            np.moveaxis(swissimage_data, 0, -1),
            extent=(swissimage_bounds.left, swissimage_bounds.right, swissimage_bounds.bottom, swissimage_bounds.top),
            interpolation="none",
            zorder=0,
            alpha=0.9,
        )

        # Overlay transparent and masked data
        plt.imshow(transparent_data, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top), cmap=ListedColormap(['none']), interpolation="none", zorder=1)
        plt.imshow(masked_data, cmap=cmap, norm=norm, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top), interpolation="none", zorder=2)

        # Overlay catchment boundaries
        catchments.boundary.plot(ax=plt.gca(), edgecolor="black", linewidth=0.7, zorder=3, label="Catchment Boundary")

        # Customize plot
        plt.xlim(dem_bounds.left, dem_bounds.right)
        plt.ylim(dem_bounds.bottom, dem_bounds.top)

        # Colorbar
        cbar = plt.colorbar(label="Water Depth (m)", boundaries=categories, ticks=[0.10, 0.25, 0.50])
        cbar.set_ticklabels(["0.10m", "0.25m", "> 0.50 m"])

        # Title and labels
        plt.title(plot_title)
        plt.xlabel("Longitude (m)")
        plt.ylabel("Latitude (m)")
        plt.legend(loc="upper right")

        # Save plot
        plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(os.path.basename(max_file))[0]}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Plot saved: {plot_filename}")

    except Exception as e:
        print(f"Failed to process {max_file}: {e}")

    print("Plot for .max file has been generated and saved.")


################################################################################################################################
import requests
from PIL import Image
from io import BytesIO
import numpy as np

def fetch_swisstopo_wms_background(bounds, pixel_size=1):
    """
    Fetches a SwissTopo WMS background image in grayscale using EPSG:2056.

    Parameters:
        bounds (rasterio.coords.BoundingBox): Bounding box in EPSG:2056.
        pixel_size (float): Desired pixel resolution in meters.

    Returns:
        np.ndarray: RGB image array of the background map.
    """
    minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)

    wms_url = "https://wms.geo.admin.ch/"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "FORMAT": "image/jpeg",
        "TRANSPARENT": "TRUE",
        "LAYERS": "ch.swisstopo.pixelkarte-grau",  # ← grayscale layer
        "STYLES": "",
        "CRS": "EPSG:2056",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": str(width),
        "HEIGHT": str(height)
    }

    try:
        response = requests.get(wms_url, params=params, timeout=15)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    except Exception as e:
        print(f"❌ Failed to fetch Swisstopo gray WMS background: {e}")
        return None
##################################################################################################################################

import matplotlib.pyplot as plt
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
from PIL import Image
from io import BytesIO
import requests

def get_swisstopo_background_image(xmin, xmax, ymin, ymax, resolution_m=2, layer='ch.swisstopo.swisstlm3d-karte-grau'):
    width_px = int((xmax - xmin) / resolution_m)
    height_px = int((ymax - ymin) / resolution_m)
    bbox = f"{xmin},{ymin},{xmax},{ymax}"
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": layer,
        "BBOX": bbox,
        "CRS": "EPSG:2056",
        "WIDTH": width_px,
        "HEIGHT": height_px,
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE"
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/png,image/*,*/*;q=0.8"
    }
    response = requests.get("https://wms.geo.admin.ch/", params=params, headers=headers)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print("❌ Failed to fetch WMS:", response.status_code)
        return None


################################################################################################################################

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def add_swisstopo_cartopy_wms_background(ax, extent, layer='ch.swisstopo.swisstlm3d-karte-grau', zorder=0):
    """
    Adds a swisstopo WMS background to the given Cartopy axis.

    Parameters:
        ax: A Cartopy GeoAxes instance.
        extent: [xmin, xmax, ymin, ymax] in EPSG:2056.
        layer: WMS layer name.
        zorder: Drawing order.
    """
    swiss_proj = ccrs.epsg(2056)
    ax.set_extent(extent, crs=swiss_proj)
    wms_url = 'https://wms.geo.admin.ch/?'
    ax.add_wms(wms_url, layers=[layer], zorder=zorder)


#####################################################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd

def g_plots_from_wd_swissTLMgray(dem_file, wd_folder, plot_output_folder, geo_ezgg_2km_ge, 
                                 location_name, rain_intensity, plot_title_prefix, 
                                 color1="violet", color2="mediumvioletred", color3="darkmagenta"):
    os.makedirs(plot_output_folder, exist_ok=True)

    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        mask = dem_data != dem_nodata_value

    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")

    # Get basemap using the original function
    basemap_img = get_swisstopo_background_image(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top)
    if basemap_img is None:
        print("⚠️ Background map not loaded.")
        return

    wd_files = sorted([os.path.join(wd_folder, f) for f in os.listdir(wd_folder) if f.endswith(".wd")])

    for i, wd_file in enumerate(wd_files):
        try:
            with rasterio.open(wd_file) as src_wd:
                wd_data = src_wd.read(1)
                wd_transform = src_wd.transform

            aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=wd_data,
                destination=aligned_data,
                src_transform=wd_transform,
                src_crs="EPSG:2056",
                dst_transform=dem_transform,
                dst_crs="EPSG:2056",
                resampling=Resampling.nearest,
            )

            masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
            transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

            categories = [0.10, 0.25, 0.50, 0.60]
            colors = [color1, color2, color3]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(categories, cmap.N, clip=True)

            fig, ax = plt.subplots(figsize=(12, 10))

            ax.imshow(basemap_img, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      interpolation="none", zorder=0)

            ax.imshow(transparent_data, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      cmap=ListedColormap(['none']), interpolation="none", zorder=1)
            ax.imshow(masked_data, cmap=cmap, norm=norm,
                      extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      interpolation="none", zorder=2)

            catchments.boundary.plot(ax=ax, edgecolor="black", linewidth=0.7, zorder=3, label="Catchment Boundary")

            ax.set_xlim(dem_bounds.left, dem_bounds.right)
            ax.set_ylim(dem_bounds.bottom, dem_bounds.top)

            cbar = plt.colorbar(ax.imshow(masked_data, cmap=cmap, norm=norm),
                                ax=ax, boundaries=categories, ticks=[0.10, 0.25, 0.50])
            cbar.set_label("Water Depth (m)", fontsize=16)
            cbar.ax.tick_params(labelsize=14)

            time_minutes = i * 5
            ax.set_title(f"{location_name} ({rain_intensity}) - {time_minutes} minutes", fontsize=18, fontweight="bold")
            ax.set_xlabel("Longitude (m)", fontsize=16)
            ax.set_ylabel("Latitude (m)", fontsize=16)
            ax.legend(loc="upper right", fontsize=14)

            plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(os.path.basename(wd_file))[0]}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"✅ Plot saved: {plot_filename}")

        except Exception as e:
            print(f"❌ Failed to process {wd_file}: {e}")

    print("✅ All plots have been generated and saved.")


###########################################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd


def g_plots_from_wd_swissTLMgray_v2(dem_file, wd_folder, plot_output_folder, geo_ezgg_2km_ge, 
                                 location_name, rain_intensity, plot_title_prefix, 
                                 color1="violet", color2="mediumvioletred", color3="darkmagenta"):
    """
    Generates plots of water depth data over a Swisstopo WMS basemap using a DEM grid.

    Parameters:
        dem_file (str): Path to the DEM file.
        wd_folder (str): Path to the folder containing .wd files.
        plot_output_folder (str): Path to save the plots.
        geo_ezgg_2km_ge (str): Path to the catchment shapefile.
        location_name (str): Name of the location to be used in the title (e.g., "Salavaux").
        rain_intensity (str): Rain intensity in mm/h to be used in the title (e.g., "25 mm/h").
        plot_title_prefix (str): Prefix for plot titles.
        color1 (str): First color (default: "violet").
        color2 (str): Second color (default: "mediumvioletred").
        color3 (str): Third color (default: "darkmagenta").
    """
    # Ensure the output directory exists
    os.makedirs(plot_output_folder, exist_ok=True)

    # Step 1: Read the DEM grid structure and mask
    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        mask = dem_data != dem_nodata_value

    # Step 2: Read the catchment shapefile
    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")

    # Step 3: Get Swisstopo WMS Map for Background using the helper function
    basemap_img = fetch_swisstopo_wms_background(dem_bounds)
    if basemap_img is None:
        return  # Stop execution if WMS fetch failed

    # Step 4: Iterate through all .wd files in the folder
    wd_files = sorted([os.path.join(wd_folder, f) for f in os.listdir(wd_folder) if f.endswith(".wd")])

    for i, wd_file in enumerate(wd_files):
        try:
            # Read water depth data
            with rasterio.open(wd_file) as src_wd:
                wd_data = src_wd.read(1)
                wd_transform = src_wd.transform

            # Reproject water depth data to match DEM grid
            aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=wd_data,
                destination=aligned_data,
                src_transform=wd_transform,
                src_crs="EPSG:2056",
                dst_transform=dem_transform,
                dst_crs="EPSG:2056",
                resampling=Resampling.nearest,
            )

            # Mask and categorize data
            masked_data = np.where((mask & (aligned_data >= 0.05)), aligned_data, np.nan)
            transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.05), 1, np.nan)

            categories = [0.05, 0.10, 0.25, 2]  # Use np.inf for all values above 0.25
            colors = [color1, color2, color3]  # User-defined colors
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(categories, cmap.N)

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))

            # Add Swisstopo WMS as background
            ax.imshow(basemap_img, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      interpolation="none", zorder=0)

            # Overlay transparent and masked data
            ax.imshow(transparent_data, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      cmap=ListedColormap(['none']), interpolation="none", zorder=1)
            ax.imshow(masked_data, cmap=cmap, norm=norm,
                      extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      interpolation="none", zorder=2)

            # Overlay catchment boundaries
            catchments.boundary.plot(ax=ax, edgecolor="black", linewidth=0.7, zorder=3, label="Catchment Boundary")

            # Customize plot
            ax.set_xlim(dem_bounds.left, dem_bounds.right)
            ax.set_ylim(dem_bounds.bottom, dem_bounds.top)

            # Colorbar
            cbar = plt.colorbar(ax.imshow(masked_data, cmap=cmap, norm=norm),
                    ax=ax, boundaries=categories, ticks=[0.05, 0.10, 0.25])

            # Increase the font size of the colorbar label
            cbar.set_label("Water Depth (m)", fontsize=16)

            # Increase font size of tick labels
            cbar.ax.tick_params(labelsize=14)  # Adjust tick labels separately

            # Dynamic Title
            time_minutes = i * 5
            ax.set_title(f"{location_name} ({rain_intensity}) - {time_minutes} minutes", fontsize=18, fontweight="bold")

            # Increase font size for axis labels
            ax.set_xlabel("Longitude (m)", fontsize=16)
            ax.set_ylabel("Latitude (m)", fontsize=16)

            # Increase font size of the legend
            ax.legend(loc="upper right", fontsize=14)

            # Save plot
            plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(os.path.basename(wd_file))[0]}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Plot saved: {plot_filename}")

        except Exception as e:
            print(f"Failed to process {wd_file}: {e}")

    print("All plots have been generated and saved.")

##########################################################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd

def g2_plots_from_wd_swissTLMgray(
    dem_file,
    wd_folder,
    plot_output_folder,
    geo_ezgg_2km_ge,
    location_name,
    rain_intensity,
    plot_title_prefix,
    color1="violet",
    color2="mediumvioletred",
    color3="darkmagenta",
    camp_polygon_path=None,
    zoom_to_camp=False
):
    """
    Generates plots of water depth data over a Swisstopo WMS basemap using a DEM grid.

    Parameters:
        dem_file (str): Path to the DEM file.
        wd_folder (str): Path to the folder containing .wd files.
        plot_output_folder (str): Path to save the plots.
        geo_ezgg_2km_ge (str): Path to the catchment shapefile.
        location_name (str): Name of the location to be used in the title (e.g., "Salavaux").
        rain_intensity (str): Rain intensity in mm/h to be used in the title (e.g., "25 mm/h").
        plot_title_prefix (str): Prefix for plot titles.
        color1, color2, color3 (str): Colors for water depth categories.
        camp_polygon_path (str): Path to campground polygon GeoJSON (optional).
        zoom_to_camp (bool): If True, zoom to the campground polygon.
    """

    os.makedirs(plot_output_folder, exist_ok=True)

    # Step 1: Read DEM
    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        mask = dem_data != dem_nodata_value

    # Step 2: Read catchments
    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")

    # Step 3: Read campground polygon (if given)
    camp_gdf = None
    if camp_polygon_path:
        try:
            camp_gdf = gpd.read_file(camp_polygon_path).to_crs("EPSG:2056")
        except Exception as e:
            print(f" Failed to read campground polygon: {e}")

    # Step 4: Fetch Swisstopo background
    basemap_img = fetch_swisstopo_wms_background(dem_bounds)
    if basemap_img is None:
        return

    # Step 5: Loop through .wd files
    wd_files = sorted([os.path.join(wd_folder, f) for f in os.listdir(wd_folder) if f.endswith(".wd")])

    for i, wd_file in enumerate(wd_files):
        try:
            with rasterio.open(wd_file) as src_wd:
                wd_data = src_wd.read(1)
                wd_transform = src_wd.transform

            aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=wd_data,
                destination=aligned_data,
                src_transform=wd_transform,
                src_crs="EPSG:2056",
                dst_transform=dem_transform,
                dst_crs="EPSG:2056",
                resampling=Resampling.nearest,
            )

            # Categorize water depth
            masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
            transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

            categories = [0.10, 0.25, 0.50, 0.60]
            colors = [color1, color2, color3]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(categories, cmap.N, clip=True)

            fig, ax = plt.subplots(figsize=(12, 10))

            # Add background map
            ax.imshow(basemap_img, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      interpolation="none", zorder=0)

            # Add water depth overlays
            ax.imshow(transparent_data, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      cmap=ListedColormap(['none']), interpolation="none", zorder=1)
            ax.imshow(masked_data, cmap=cmap, norm=norm,
                      extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      interpolation="none", zorder=2)

            # Plot catchment
            catchments.boundary.plot(ax=ax, edgecolor="black", linewidth=0.7, zorder=3, label="Catchment Boundary")

            # Plot campground outline if available
            if camp_gdf is not None:
                camp_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=1.2, zorder=4, label="Campground")
                if zoom_to_camp:
                    bounds = camp_gdf.total_bounds
                    ax.set_xlim(bounds[0] - 10, bounds[2] + 10)
                    ax.set_ylim(bounds[1] - 10, bounds[3] + 10)
                else:
                    ax.set_xlim(dem_bounds.left, dem_bounds.right)
                    ax.set_ylim(dem_bounds.bottom, dem_bounds.top)
            else:
                ax.set_xlim(dem_bounds.left, dem_bounds.right)
                ax.set_ylim(dem_bounds.bottom, dem_bounds.top)

            # Colorbar
            cbar = plt.colorbar(ax.imshow(masked_data, cmap=cmap, norm=norm),
                                ax=ax, boundaries=categories, ticks=[0.10, 0.25, 0.50])
            cbar.set_label("Water Depth (m)", fontsize=16)
            cbar.ax.tick_params(labelsize=14)

            # Title and labels
            time_minutes = i * 5
            ax.set_title(f"{plot_title_prefix} - {location_name} ({rain_intensity}) - {time_minutes} min",
                         fontsize=18, fontweight="bold")
            ax.set_xlabel("Longitude (m)", fontsize=16)
            ax.set_ylabel("Latitude (m)", fontsize=16)
            ax.legend(loc="upper right", fontsize=14)

            # Save plot
            plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(os.path.basename(wd_file))[0]}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f" Plot saved: {plot_filename}")

        except Exception as e:
            print(f"Failed to process {wd_file}: {e}")

    print(" All plots have been generated and saved.")

#######################################################################################################################
############################################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd

def g_plot_maxwd_swissTLM(
    dem_file, 
    max_file, 
    plot_output_folder, 
    geo_ezgg_2km_ge, 
    location_name, 
    rain_intensity, 
    color1="violet", color2="mediumvioletred", color3="darkmagenta"
):
    os.makedirs(plot_output_folder, exist_ok=True)

    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        mask = dem_data != dem_nodata_value

    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")

    basemap_img = get_swisstopo_background_image(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top)
    if basemap_img is None:
        print("Failed to fetch Swisstopo WMS basemap.")
        return

    try:
        with rasterio.open(max_file) as src_max:
            max_data = src_max.read(1)
            max_transform = src_max.transform

        aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
        reproject(
            source=max_data,
            destination=aligned_data,
            src_transform=max_transform,
            src_crs="EPSG:2056",
            dst_transform=dem_transform,
            dst_crs="EPSG:2056",
            resampling=Resampling.nearest,
        )

        masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
        transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

        categories = [0.10, 0.25, 0.50, 0.60]
        colors = [color1, color2, color3]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(categories, cmap.N, clip=True)

        fig, ax = plt.subplots(figsize=(12, 10))

        ax.imshow(basemap_img, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                  interpolation="none", zorder=0)
        ax.imshow(transparent_data, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                  cmap=ListedColormap(['none']), interpolation="none", zorder=1)
        ax.imshow(masked_data, cmap=cmap, norm=norm,
                  extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                  interpolation="none", zorder=2)

        catchments.boundary.plot(ax=ax, edgecolor="black", linewidth=0.7, zorder=3, label="Catchment Boundary")

        ax.set_xlim(dem_bounds.left, dem_bounds.right)
        ax.set_ylim(dem_bounds.bottom, dem_bounds.top)

        cbar = plt.colorbar(ax.imshow(masked_data, cmap=cmap, norm=norm),
                            ax=ax, boundaries=categories, ticks=[0.10, 0.25, 0.50])
        cbar.set_label("Water Depth (m)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        ax.set_title(f"{location_name} ({rain_intensity}) - Max Water Depth", fontsize=18, fontweight="bold")
        ax.set_xlabel("Longitude (m)", fontsize=16)
        ax.set_ylabel("Latitude (m)", fontsize=16)
        ax.legend(loc="upper right", fontsize=14)

        plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(os.path.basename(max_file))[0]}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\u2705 Plot saved: {plot_filename}")

    except Exception as e:
        print(f"\u274c Failed to process {max_file}: {e}")

    print("\u2705 Plot for .max file has been generated and saved.")


############################################################################################
############ COMPARING SOLVERS JUST FOR ONE SCENARIO#######################################
###########################################################################################
######### crs first make sure that it is right for Switzerland ###########################
########################################################################################
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
import os

def compare_solver_extent_map(
    dem_file,
    solver_a_file,
    solver_b_file,
    output_folder,
    label_a="Solver A",
    label_b="Solver B",
    location="Location",
    rain="Rain X mm/h",
    threshold=0.10
):
    """
    Creates a categorical flood extent difference map comparing two solvers (A and B).
    """

    os.makedirs(output_folder, exist_ok=True)

    # Load DEM
    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_transform = src_dem.transform
        dem_shape = dem_data.shape
        dem_bounds = src_dem.bounds
        dem_nodata = src_dem.nodata
        mask = dem_data != dem_nodata
        extent = (dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top)

    # Helper to align .max files
    def align_max(file):
        with rasterio.open(file) as src:
            data = src.read(1)
            src_transform = src.transform

        aligned = np.full(dem_shape, np.nan, dtype=np.float32)
        reproject(
            source=data,
            destination=aligned,
            src_transform=src_transform,
            src_crs="EPSG:2056",
            dst_transform=dem_transform,
            dst_crs="EPSG:2056",
            resampling=Resampling.nearest
        )

        return np.where(mask, aligned, np.nan)

    # Read and threshold both solvers
    a_aligned = align_max(solver_a_file)
    b_aligned = align_max(solver_b_file)

    flood_a = (a_aligned >= threshold)
    flood_b = (b_aligned >= threshold)

    # 0 = dry in both → white
    # 1 = flooded in both → green
    # 2 = overpredicted by A → blue
    # 3 = overpredicted by B → red
    comparison = np.full(dem_shape, np.nan)
    comparison[~mask] = np.nan
    comparison[(~flood_a) & (~flood_b)] = 0  # white
    comparison[(flood_a) & (flood_b)] = 1    # green
    comparison[(flood_a) & (~flood_b)] = 2   # blue
    comparison[(~flood_a) & (flood_b)] = 3   # red

    # Colors: white, green, blue, red
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", "green", "blue", "red"])
    labels = ["Dry in both", "Flood in both", f"Only {label_a}", f"Only {label_b}"]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(comparison, cmap=cmap, extent=extent, interpolation="none")
    ax.set_title(f"{location} – {rain}\nFlood Extent Comparison\n{label_a} vs {label_b}", fontsize=14)
    ax.axis("off")

    # Custom legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=cmap(i), label=labels[i]) for i in range(4)]
    ax.legend(handles=patches, loc="lower right", fontsize=10)

    # Save
    safe_rain = re.sub(r"\D", "", rain) + "mmhr"
    out_path = os.path.join(output_folder, f"{location}_{label_a}_vs_{label_b}_{safe_rain}.png")
    plt.savefig(out_path, dpi=800, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {out_path}")


####################################################################################################
#####################FOR LIESTAL ###################################################################
####################FORECAST ######################################################################
###################################################################################################

from datetime import datetime, timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd

def g_plots_selected_wd_liestal(dem_file, wd_folder, plot_output_folder, geo_ezgg_2km_ge, 
                                      plot_title_prefix,
                                      initial_datetime_str, lead_times_hours,
                                      color1="violet", color2="mediumvioletred", color3="darkmagenta"):
    """
    Plots specific .wd files for the Liestal case using Swisstopo background.
    Each plot gets a title like "Liestal – 2024-06-25T15:00:00 + X hour lead time".
    """

    # Specific .wd filenames and assumed order matching the lead times
    selected_filenames = [
        "Liestal_2m_1_1-0012.wd",
        "Liestal_2m_1_1-0024.wd",
        "Liestal_2m_1_1-0036.wd",
        "Liestal_2m_1_1-0048.wd",
        "Liestal_2m_1_1-0060.wd"
    ]

    # Parse base datetime
    base_time = datetime.strptime(initial_datetime_str, "%Y-%m-%dT%H:%M:%S")

    # Ensure output directory exists
    os.makedirs(plot_output_folder, exist_ok=True)

    # Load DEM
    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        mask = dem_data != dem_nodata_value

    # Load catchments
    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")

    # Get background map
    basemap_img = fetch_swisstopo_wms_background(dem_bounds)
    if basemap_img is None:
        print("Failed to fetch basemap.")
        return

    # Plot selected files
    for filename, lead_hours in zip(selected_filenames, lead_times_hours):
        wd_file_path = os.path.join(wd_folder, filename)
        if not os.path.isfile(wd_file_path):
            print(f"File not found: {filename}")
            continue

        try:
            with rasterio.open(wd_file_path) as src_wd:
                wd_data = src_wd.read(1)
                wd_transform = src_wd.transform

            aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=wd_data,
                destination=aligned_data,
                src_transform=wd_transform,
                src_crs="EPSG:2056",
                dst_transform=dem_transform,
                dst_crs="EPSG:2056",
                resampling=Resampling.nearest,
            )

            masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
            transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

            categories = [0.10, 0.25, 0.50, 0.60]
            colors = [color1, color2, color3]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(categories, cmap.N, clip=True)

            fig, ax = plt.subplots(figsize=(12, 10))

            ax.imshow(basemap_img, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      interpolation="none", zorder=0)
            ax.imshow(transparent_data, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      cmap=ListedColormap(['none']), interpolation="none", zorder=1)
            ax.imshow(masked_data, cmap=cmap, norm=norm,
                      extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      interpolation="none", zorder=2)

            catchments.boundary.plot(ax=ax, edgecolor="black", linewidth=0.7, zorder=3, label="Catchment Boundary")

            ax.set_xlim(dem_bounds.left, dem_bounds.right)
            ax.set_ylim(dem_bounds.bottom, dem_bounds.top)

            cbar = plt.colorbar(ax.imshow(masked_data, cmap=cmap, norm=norm),
                                ax=ax, boundaries=categories, ticks=[0.10, 0.25, 0.50])
            cbar.set_label("Water Depth (m)", fontsize=16)
            cbar.ax.tick_params(labelsize=14)

            # Format title
            title = f"{plot_title_prefix} – {initial_datetime_str} + {lead_hours} hour lead time"
            ax.set_title(title, fontsize=18, fontweight="bold")

            ax.set_xlabel("Longitude (m)", fontsize=16)
            ax.set_ylabel("Latitude (m)", fontsize=16)
            ax.legend(loc="upper right", fontsize=14)

            plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Plot saved: {plot_filename}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print("Selected plots have been generated and saved.")

#####################################################################################################

#######################################################################################

from datetime import datetime
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd

def g_plots_selected_wd_liestal_dinamic(dem_file, wd_folder, plot_output_folder, geo_ezgg_2km_ge, 
                                        plot_title_prefix,
                                        initial_datetime_str, lead_times_hours,
                                        color1="violet", color2="mediumvioletred", color3="darkmagenta",
                                        xlim=None, ylim=None):
    """
    Dynamically plots water depth files for the Liestal case using Swisstopo background.
    Automatically infers ensemble member number from folder name.

    Parameters:
        dem_file (str): Path to DEM file.
        wd_folder (str): Path to folder with .wd files.
        plot_output_folder (str): Folder to save output plots.
        geo_ezgg_2km_ge (str): Catchment file in EPSG:2056.
        plot_title_prefix (str): Title prefix for plots (e.g., "Liestal").
        initial_datetime_str (str): Start time in "YYYY-MM-DDTHH:MM:SS".
        lead_times_hours (list[int]): Lead times for each plot (e.g., [1,2,3,4,5]).
        color1, color2, color3 (str): Color definitions.
        xlim, ylim (tuple): Optional zoom limits (EPSG:2056).
    """

    #  Extract ensemble number from folder name (e.g., Liestal_2m_2_fv1-gpu → 2)
    folder_name = os.path.basename(os.path.normpath(wd_folder))
    match = re.search(r"Liestal_2m_(\d+)_", folder_name)
    if not match:
        raise ValueError(f"Could not extract ensemble number from folder name: {folder_name}")
    ensemble_number = match.group(1)

    #  Build filenames dynamically
    time_steps = ["0012", "0024", "0036", "0048", "0060","0072","0084","0096","0108","0120"]
    selected_filenames = [f"Liestal_2m_{ensemble_number}_{ensemble_number}-{t}.wd" for t in time_steps]

    #  Parse base datetime
    base_time = datetime.strptime(initial_datetime_str, "%Y-%m-%dT%H:%M:%S")

    os.makedirs(plot_output_folder, exist_ok=True)

    #  Load DEM and metadata
    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        mask = dem_data != dem_nodata_value

    # Load catchment boundaries
    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")

    #  Fetch high-res background map
    basemap_img = fetch_swisstopo_wms_background(dem_bounds, pixel_size=1)   
    if basemap_img is None:
        print("Failed to fetch basemap.")
        return

    #  Plot each .wd file
    for filename, lead_hours in zip(selected_filenames, lead_times_hours):
        wd_file_path = os.path.join(wd_folder, filename)
        if not os.path.isfile(wd_file_path):
            print(f"File not found: {filename}")
            continue

        try:
            with rasterio.open(wd_file_path) as src_wd:
                wd_data = src_wd.read(1)
                wd_transform = src_wd.transform

            aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=wd_data,
                destination=aligned_data,
                src_transform=wd_transform,
                src_crs="EPSG:2056",
                dst_transform=dem_transform,
                dst_crs="EPSG:2056",
                resampling=Resampling.nearest,
            )

            masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
            transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

            categories = [0.10, 0.25, 0.50, 0.60]
            colors = [color1, color2, color3]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(categories, cmap.N, clip=True)

            fig, ax = plt.subplots(figsize=(12, 10))

            ax.imshow(basemap_img, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      interpolation="none", zorder=0)
            ax.imshow(transparent_data, extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      cmap=ListedColormap(['none']), interpolation="none", zorder=1)
            ax.imshow(masked_data, cmap=cmap, norm=norm,
                      extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
                      interpolation="none", zorder=2)

            catchments.boundary.plot(ax=ax, edgecolor="black", linewidth=0.7, zorder=3, label="Catchment Boundary")

            #  Zoom
            ax.set_xlim(xlim if xlim else (dem_bounds.left, dem_bounds.right))
            ax.set_ylim(ylim if ylim else (dem_bounds.bottom, dem_bounds.top))

            #  Colorbar
            cbar = plt.colorbar(ax.imshow(masked_data, cmap=cmap, norm=norm),
                                ax=ax, boundaries=categories, ticks=[0.10, 0.25, 0.50])
            cbar.set_label("Water Depth (m)", fontsize=16)
            cbar.ax.tick_params(labelsize=14)

            #  Title and labels
            title = f"{plot_title_prefix} – {initial_datetime_str} + {lead_hours} hour lead time"
            ax.set_title(title, fontsize=18, fontweight="bold")
            ax.set_xlabel("Longitude (m)", fontsize=16)
            ax.set_ylabel("Latitude (m)", fontsize=16)
            ax.legend(loc="upper right", fontsize=14)

            #  Save figure
            plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(plot_filename, dpi=1000, bbox_inches="tight")
            plt.close()

            print(f"Plot saved: {plot_filename}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print(" All selected plots have been generated and saved.")

##############################################################################################################################

from datetime import datetime
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.plot import plotting_extent
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
import cartopy.crs as ccrs


def g_plots_selected_wd_liestal_dinamic_no_cbar(dem_file, wd_folder, plot_output_folder, geo_ezgg_2km_ge, 
                                                 plot_title_prefix,
                                                 initial_datetime_str, lead_times_hours,
                                                 color1="violet", color2="mediumvioletred", color3="darkmagenta",
                                                 xlim=None, ylim=None):
    """
    Plots water depth maps without colorbar for animation/video use.
    """
    import requests
    from PIL import Image
    from io import BytesIO
    import matplotlib.pyplot as plt
    import rasterio
    import numpy as np
    from rasterio.warp import reproject, Resampling
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import geopandas as gpd
    import cartopy.crs as ccrs
    import os, re
    from datetime import datetime

    def get_swisstopo_background_image(xmin, xmax, ymin, ymax, resolution_m=2, layer='ch.swisstopo.swisstlm3d-karte-grau'):
        width_px = int((xmax - xmin) / resolution_m)
        height_px = int((ymax - ymin) / resolution_m)
        bbox = f"{xmin},{ymin},{xmax},{ymax}"
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "BBOX": bbox,
            "CRS": "EPSG:2056",
            "WIDTH": width_px,
            "HEIGHT": height_px,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE"
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/png,image/*,*/*;q=0.8"
        }
        response = requests.get("https://wms.geo.admin.ch/", params=params, headers=headers)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(" Failed to fetch WMS:", response.status_code)
            return None

    folder_name = os.path.basename(os.path.normpath(wd_folder))
    match = re.search(r"Liestal_2m_(\d+)", folder_name)
    if not match:
        raise ValueError(f"Could not extract ensemble number from folder name: {folder_name}")
    ensemble_number = match.group(1)

    time_steps = [f"{h*12:04d}" for h in lead_times_hours]
    selected_filenames = [f"Liestal_2m_{ensemble_number}_{ensemble_number}-{t}.wd" for t in time_steps]
    base_time = datetime.strptime(initial_datetime_str, "%Y-%m-%dT%H:%M:%S")
    os.makedirs(plot_output_folder, exist_ok=True)

    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        extent = (dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top)
        mask = dem_data != dem_nodata_value

    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")
    xlim = xlim if xlim else (extent[0], extent[1])
    ylim = ylim if ylim else (extent[2], extent[3])
    zoom_extent = (xlim[0], xlim[1], ylim[0], ylim[1])

    for filename, lead_hours in zip(selected_filenames, lead_times_hours):
        wd_file_path = os.path.join(wd_folder, filename)
        if not os.path.isfile(wd_file_path):
            print(f"File not found: {filename}")
            continue

        try:
            with rasterio.open(wd_file_path) as src_wd:
                wd_data = src_wd.read(1)
                wd_transform = src_wd.transform

            aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=wd_data,
                destination=aligned_data,
                src_transform=wd_transform,
                src_crs="EPSG:2056",
                dst_transform=dem_transform,
                dst_crs="EPSG:2056",
                resampling=Resampling.nearest,
            )

            masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
            transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

            categories = [0.10, 0.25, 0.50, 0.60]
            colors = [color1, color2, color3]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(categories, cmap.N, clip=True)

            fig = plt.figure(figsize=(12, 10))
            crs_2056 = ccrs.epsg(2056)
            ax = fig.add_subplot(1, 1, 1, projection=crs_2056)
            ax.set_extent(zoom_extent, crs=crs_2056)

            bg_img = get_swisstopo_background_image(*zoom_extent, resolution_m=2)
            if bg_img is not None:
                ax.imshow(bg_img, extent=zoom_extent, transform=crs_2056, zorder=0)
            else:
                print(" Background image not loaded.")

            ax.imshow(transparent_data, extent=extent, transform=crs_2056,
                      cmap=ListedColormap(['none']), interpolation="none", zorder=1)
            ax.imshow(masked_data, extent=extent, transform=crs_2056,
                      cmap=cmap, norm=norm, interpolation="none", zorder=2)

            catchments.boundary.plot(ax=ax, edgecolor="black", linewidth=0.7, zorder=3)

            title = f"{plot_title_prefix} – {initial_datetime_str} + {lead_hours} hour lead time"
            ax.set_title(title, fontsize=18, fontweight="bold")
            ax.set_xlabel("Easting (m)", fontsize=16)
            ax.set_ylabel("Northing (m)", fontsize=16)

            plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(filename)[0]}_nocbar.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f" Plot saved: {plot_filename}")

        except Exception as e:
            print(f" Failed to process {filename}: {e}")

    print(" All video-ready plots (without colorbar) have been generated and saved.")

######################################################################################################

def g_plots_selected_wd_liestal_dinamic_no_cbar(
    dem_file, wd_folder, plot_output_folder, geo_ezgg_2km_ge,
    plot_title_prefix, initial_datetime_str, lead_times_hours,
    color1="violet", color2="mediumvioletred", color3="darkmagenta",
    xlim=None, ylim=None
):
    """
    Plots deterministic water depth maps from observation/single scenario (no ensemble index) without colorbar.
    Example filename: Liestal_2m_0000.wd, Liestal_2m_0012.wd, etc.
    """
    import requests
    from PIL import Image
    from io import BytesIO
    import matplotlib.pyplot as plt
    import rasterio
    import numpy as np
    from rasterio.warp import reproject, Resampling
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import geopandas as gpd
    import cartopy.crs as ccrs
    import os
    from datetime import datetime

    def get_swisstopo_background_image(xmin, xmax, ymin, ymax, resolution_m=2, layer='ch.swisstopo.swisstlm3d-karte-grau'):
        width_px = int((xmax - xmin) / resolution_m)
        height_px = int((ymax - ymin) / resolution_m)
        bbox = f"{xmin},{ymin},{xmax},{ymax}"
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "BBOX": bbox,
            "CRS": "EPSG:2056",
            "WIDTH": width_px,
            "HEIGHT": height_px,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE"
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/png,image/*,*/*;q=0.8"
        }
        response = requests.get("https://wms.geo.admin.ch/", params=params, headers=headers)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(" Failed to fetch WMS:", response.status_code)
            return None

    time_steps = [f"{h*12:04d}" for h in lead_times_hours]
    selected_filenames = [f"Liestal_2m-{t}.wd" for t in time_steps]
    base_time = datetime.strptime(initial_datetime_str, "%Y-%m-%dT%H:%M:%S")
    os.makedirs(plot_output_folder, exist_ok=True)

    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        extent = (dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top)
        mask = dem_data != dem_nodata_value

    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")
    xlim = xlim if xlim else (extent[0], extent[1])
    ylim = ylim if ylim else (extent[2], extent[3])
    zoom_extent = (xlim[0], xlim[1], ylim[0], ylim[1])

    for filename, lead_hours in zip(selected_filenames, lead_times_hours):
        wd_file_path = os.path.join(wd_folder, filename)
        if not os.path.isfile(wd_file_path):
            print(f"File not found: {filename}")
            continue

        try:
            with rasterio.open(wd_file_path) as src_wd:
                wd_data = src_wd.read(1)
                wd_transform = src_wd.transform

            aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=wd_data,
                destination=aligned_data,
                src_transform=wd_transform,
                src_crs="EPSG:2056",
                dst_transform=dem_transform,
                dst_crs="EPSG:2056",
                resampling=Resampling.nearest,
            )

            masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
            transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

            categories = [0.10, 0.25, 0.50, 0.60]
            colors = [color1, color2, color3]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(categories, cmap.N, clip=True)

            fig = plt.figure(figsize=(12, 10))
            crs_2056 = ccrs.epsg(2056)
            ax = fig.add_subplot(1, 1, 1, projection=crs_2056)
            ax.set_extent(zoom_extent, crs=crs_2056)

            bg_img = get_swisstopo_background_image(*zoom_extent, resolution_m=2)
            if bg_img is not None:
                ax.imshow(bg_img, extent=zoom_extent, transform=crs_2056, zorder=0)
            else:
                print(" Background image not loaded.")

            ax.imshow(transparent_data, extent=extent, transform=crs_2056,
                      cmap=ListedColormap(['none']), interpolation="none", zorder=1)
            ax.imshow(masked_data, extent=extent, transform=crs_2056,
                      cmap=cmap, norm=norm, interpolation="none", zorder=2)

            catchments.boundary.plot(ax=ax, edgecolor="black", linewidth=0.7, zorder=3)

            title = f"{plot_title_prefix} – {initial_datetime_str} + {lead_hours} hour lead time"
            ax.set_title(title, fontsize=18, fontweight="bold")
            ax.set_xlabel("Easting (m)", fontsize=16)
            ax.set_ylabel("Northing (m)", fontsize=16)

            plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(filename)[0]}_nocbar.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"✔ Plot saved: {plot_filename}")

        except Exception as e:
            print(f" Failed to process {filename}: {e}")

    print(" All deterministic plots (no colorbar) have been generated and saved.")

############################################################################################

def plot_Liestal_Combiprecip_perhour(
    dem_file, wd_folder, plot_output_folder, geo_ezgg_2km_ge,
    plot_title_prefix, initial_datetime_str, lead_times_hours,
    color1="violet", color2="mediumvioletred", color3="darkmagenta",
    xlim=None, ylim=None
):
    """
    Plots deterministic water depth maps from observation/single scenario (no ensemble index) without colorbar.
    Example filename: Liestal_2m-0000.wd, Liestal_2m-0001.wd, etc.
    """
    import requests
    from PIL import Image
    from io import BytesIO
    import matplotlib.pyplot as plt
    import rasterio
    import numpy as np
    from rasterio.warp import reproject, Resampling
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import geopandas as gpd
    import cartopy.crs as ccrs
    import os
    from datetime import datetime, timedelta

    def get_swisstopo_background_image(xmin, xmax, ymin, ymax, resolution_m=2, layer='ch.swisstopo.swisstlm3d-karte-grau'):
        width_px = int((xmax - xmin) / resolution_m)
        height_px = int((ymax - ymin) / resolution_m)
        bbox = f"{xmin},{ymin},{xmax},{ymax}"
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "BBOX": bbox,
            "CRS": "EPSG:2056",
            "WIDTH": width_px,
            "HEIGHT": height_px,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE"
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/png,image/*,*/*;q=0.8"
        }
        response = requests.get("https://wms.geo.admin.ch/", params=params, headers=headers)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(" Failed to fetch WMS:", response.status_code)
            return None

    time_steps = [f"{h:04d}" for h in lead_times_hours]
    selected_filenames = [f"Liestal_2m-{t}.wd" for t in time_steps]
    base_time = datetime.strptime(initial_datetime_str, "%Y-%m-%dT%H:%M:%S")
    os.makedirs(plot_output_folder, exist_ok=True)

    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        extent = (dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top)
        mask = dem_data != dem_nodata_value

    catchments = gpd.read_file(geo_ezgg_2km_ge).to_crs("EPSG:2056")
    xlim = xlim if xlim else (extent[0], extent[1])
    ylim = ylim if ylim else (extent[2], extent[3])
    zoom_extent = (xlim[0], xlim[1], ylim[0], ylim[1])

    for filename, lead_hours in zip(selected_filenames, lead_times_hours):
        wd_file_path = os.path.join(wd_folder, filename)
        if not os.path.isfile(wd_file_path):
            print(f"File not found: {filename}")
            continue

        try:
            with rasterio.open(wd_file_path) as src_wd:
                wd_data = src_wd.read(1)
                wd_transform = src_wd.transform

            aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=wd_data,
                destination=aligned_data,
                src_transform=wd_transform,
                src_crs="EPSG:2056",
                dst_transform=dem_transform,
                dst_crs="EPSG:2056",
                resampling=Resampling.nearest,
            )

            masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
            transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

            categories = [0.10, 0.25, 0.50, 0.60]
            colors = [color1, color2, color3]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(categories, cmap.N, clip=True)

            fig = plt.figure(figsize=(12, 10))
            crs_2056 = ccrs.epsg(2056)
            ax = fig.add_subplot(1, 1, 1, projection=crs_2056)
            ax.set_extent(zoom_extent, crs=crs_2056)

            bg_img = get_swisstopo_background_image(*zoom_extent, resolution_m=2)
            if bg_img is not None:
                ax.imshow(bg_img, extent=zoom_extent, transform=crs_2056, zorder=0)
            else:
                print(" Background image not loaded.")

            ax.imshow(transparent_data, extent=extent, transform=crs_2056,
                      cmap=ListedColormap(['none']), interpolation="none", zorder=1)
            ax.imshow(masked_data, extent=extent, transform=crs_2056,
                      cmap=cmap, norm=norm, interpolation="none", zorder=2)

            catchments.boundary.plot(ax=ax, edgecolor="black", linewidth=0.7, zorder=3)

            # ⏰ Actual forecast time title
            forecast_time = base_time + timedelta(hours=lead_hours)
            title = f"{plot_title_prefix} – {forecast_time.strftime('%Y-%m-%dT%H:%M:%S')}"
            ax.set_title(title, fontsize=18, fontweight="bold")
            ax.set_xlabel("Easting (m)", fontsize=16)
            ax.set_ylabel("Northing (m)", fontsize=16)

            plot_filename = os.path.join(plot_output_folder, f"{os.path.splitext(filename)[0]}_nocbar.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"✔ Plot saved: {plot_filename}")

        except Exception as e:
            print(f" Failed to process {filename}: {e}")

    print("✅ All deterministic plots (no colorbar) have been generated and saved.")

###########################################################################################

from datetime import datetime, timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs


def plot_deterministic_perhour(
    case_name,
    dem_file,
    wd_folder,
    plot_output_folder,
    lead_times_hours,
    initial_datetime_str=None,
    forecast_times=None,   #  NEW: observational times
    color1="violet",
    color2="mediumvioletred",
    color3="darkmagenta",
    xlim=None,
    ylim=None,
):
    """
    Plots deterministic water depth maps (single scenario, no ensembles) without colorbar.

    Example filenames:
      Zell_2m-0000.wd, Zell_2m-0001.wd, ...

    Parameters
    ----------
    case_name : str
        Simulation case name, e.g. "Zell_2m".
    dem_file : str
        Path to DEM raster.
    wd_folder : str
        Folder containing .wd files.
    plot_output_folder : str
        Where PNG plots will be saved.
    lead_times_hours : list[int]
        Lead times in hours (used to find filenames).
    initial_datetime_str : str or None
        Optional start time (ISO string). Used only if forecast_times is None.
    forecast_times : list[str] or None
        Observational timestamps (ISO strings). If given, they replace lead times in titles.
    """

    # ─── WMS Background ─────────────────────────────────────────
    def get_swisstopo_background_image(xmin, xmax, ymin, ymax, resolution_m=2,
                                       layer="ch.swisstopo.swisstlm3d-karte-grau"):
        import requests
        from PIL import Image
        from io import BytesIO

        width_px = int((xmax - xmin) / resolution_m)
        height_px = int((ymax - ymin) / resolution_m)
        bbox = f"{xmin},{ymin},{xmax},{ymax}"
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "BBOX": bbox,
            "CRS": "EPSG:2056",
            "WIDTH": width_px,
            "HEIGHT": height_px,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE"
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/png,image/*,*/*;q=0.8"
        }
        response = requests.get("https://wms.geo.admin.ch/", params=params, headers=headers)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f" Failed to fetch WMS: {response.status_code}")
            return None

    # ─── File names ────────────────────────────────────────────
    time_steps = [f"{h:04d}" for h in lead_times_hours]
    selected_filenames = [f"{case_name}-{t}.wd" for t in time_steps]

    base_time = None
    if initial_datetime_str and forecast_times is None:
        base_time = datetime.strptime(initial_datetime_str, "%Y-%m-%dT%H:%M:%S")

    os.makedirs(plot_output_folder, exist_ok=True)

    # ─── DEM for extent and mask ───────────────────────────────
    with rasterio.open(dem_file) as src_dem:
        dem_data = src_dem.read(1)
        dem_nodata_value = src_dem.nodata if src_dem.nodata is not None else -9999
        dem_transform = src_dem.transform
        dem_bounds = src_dem.bounds
        dem_shape = dem_data.shape
        extent = (dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top)
        mask = dem_data != dem_nodata_value

    xlim = xlim if xlim else (extent[0], extent[1])
    ylim = ylim if ylim else (extent[2], extent[3])
    zoom_extent = (xlim[0], xlim[1], ylim[0], ylim[1])

    # ─── Loop through time steps ──────────────────────────────
    for idx, (filename, lead_hours) in enumerate(zip(selected_filenames, lead_times_hours)):
        wd_file_path = os.path.join(wd_folder, filename)
        if not os.path.isfile(wd_file_path):
            print(f" File not found: {filename}")
            continue

        try:
            with rasterio.open(wd_file_path) as src_wd:
                wd_data = src_wd.read(1)
                wd_transform = src_wd.transform

            aligned_data = np.full(dem_shape, np.nan, dtype=np.float32)
            reproject(
                source=wd_data,
                destination=aligned_data,
                src_transform=wd_transform,
                src_crs="EPSG:2056",
                dst_transform=dem_transform,
                dst_crs="EPSG:2056",
                resampling=Resampling.nearest,
            )

            masked_data = np.where((mask & (aligned_data >= 0.10)), aligned_data, np.nan)
            transparent_data = np.where((aligned_data >= 0) & (aligned_data < 0.10), 1, np.nan)

            categories = [0.10, 0.30, 0.50, 0.60]
            colors = [color1, color2, color3]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(categories, cmap.N, clip=True)

            fig = plt.figure(figsize=(12, 10))
            crs_2056 = ccrs.epsg(2056)
            ax = fig.add_subplot(1, 1, 1, projection=crs_2056)
            ax.set_extent(zoom_extent, crs=crs_2056)

            bg_img = get_swisstopo_background_image(*zoom_extent, resolution_m=2)
            if bg_img is not None:
                ax.imshow(bg_img, extent=zoom_extent, transform=crs_2056, zorder=0)

            ax.imshow(transparent_data, extent=extent, transform=crs_2056,
                      cmap=ListedColormap(['none']), interpolation="none", zorder=1)
            ax.imshow(masked_data, extent=extent, transform=crs_2056,
                      cmap=cmap, norm=norm, interpolation="none", zorder=2)

            # ─── Title handling ──────────────────────────────
            if forecast_times is not None:
                # Observational timestamps
                title = f"{case_name.replace('_2m','')} – {forecast_times[idx]}"
            elif base_time:
                # Initial datetime string (forecast mode)
                forecast_time = base_time + timedelta(hours=lead_hours)
                title = f"{case_name.replace('_2m','')} – {forecast_time.strftime('%Y-%m-%dT%H:%M:%S')}"
            else:
                # Default: simple hour labels
                title = f"{case_name.replace('_2m','')} – Hour {lead_hours}"

            ax.set_title(title, fontsize=18, fontweight="bold")
            ax.set_xlabel("Easting (m)", fontsize=16)
            ax.set_ylabel("Northing (m)", fontsize=16)

            plot_filename = os.path.join(
                plot_output_folder, f"{os.path.splitext(filename)[0]}_nocbar.png"
            )
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"✔ Plot saved: {plot_filename}")

        except Exception as e:
            print(f"✘ Failed to process {filename}: {e}")

    print(f" All deterministic plots (no colorbar) generated for {case_name}.")


################################################################################################
######################## ASCII in netcdfile ####################################################
################################################################################################
import os
import numpy as np
import xarray as xr
import rioxarray as rxr


def ascii_single_member_to_netcdf(
    *,
    out_nc: str,
    folder: str,
    base: str,
    start: int,
    end: int,
    width: int = 4,
    # map output variable names -> LISFLOOD file extensions
    var_map: dict[str, str] | None = None,
    crs: str = "EPSG:2056",
    nodata: float | None = None,
    dtype: str = "float32",
    complevel: int = 4,
    chunks: dict | None = None,
    step_regex: str = r"-(\d{4})\.(?:\w+)$",  # kept for backward compatibility (unused; robust parser used)
    strict_align: bool = True,
    use_nan_fill: bool = False,
    wet_threshold: float = 0.0,          # mask derived fields where wd < threshold
    skip_missing: bool = False,

    # ---- TIME SETTINGS ----
    dim_name: str = "REFERENCE_TS",      # dimension name in NetCDF
    reference_start: str = "2022-05-05T12:00:00.000000000",  # first timestamp
    dt_minutes: int = 60,                # next step = +dt_minutes

    realization: int | None = None,      # optional 1-length realization dim
) -> str:
    """
    Read *one* LISFLOOD member (wd, Vx, Vy, Qx, Qy) over indices [start..end],
    write a NetCDF where the time dimension is `dim_name` and is datetime64[ns].

    Time mapping:
      - The FIRST file read (index = `start`) is exactly `reference_start`
      - Each next file is +dt_minutes minutes
    """

    # -------------------- helpers --------------------

    def _open_ascii(path: str) -> xr.DataArray:
        """Open an ASCII raster (AAIGrid-style) via rioxarray, return 2D (y,x)."""
        try:
            da = rxr.open_rasterio(path, masked=True, chunks=chunks)
        except Exception:
            da = rxr.open_rasterio(path, masked=True, chunks=chunks, driver="AAIGrid")
        # rioxarray returns (band, y, x) -> squeeze to (y, x)
        return da.squeeze("band", drop=True)

    def _build_range(ext: str) -> list[str]:
        """Build file list for indices start..end inclusive."""
        return [os.path.join(folder, f"{base}-{i:0{width}d}.{ext}") for i in range(start, end + 1)]

    def _parse_step(fname: str) -> int:
        """
        Robustly parse step from filenames like:
          <base>-0000.wd
          <base>-0123.Vx
        """
        b = os.path.basename(fname)
        stem, _ = os.path.splitext(b)  # e.g. "Zell_2m_accv14-0000"
        if "-" not in stem:
            raise ValueError(f"Cannot parse timestep (no '-') from filename: {b}")
        tail = stem.split("-")[-1]
        if not tail.isdigit():
            raise ValueError(f"Cannot parse timestep (not digits) from filename: {b}")
        return int(tail)

    def _match_axis_len(arr: xr.DataArray, target_len: int, axis_name: str) -> xr.DataArray:
        """
        Pad/crop a DataArray along axis_name to match target_len.
        Handles common LISFLOOD face vs cell dimension +/-1 and general mismatch.
        """
        cur = arr.sizes.get(axis_name, None)
        if cur is None:
            raise KeyError(f"Axis '{axis_name}' not found in array dims {arr.dims}")

        if cur == target_len:
            return arr

        # If off by 1, try best-guess fix (common with face/cell grids)
        if cur == target_len - 1:
            # pad one cell at both ends (duplicate edges) then slice
            first = arr.isel({axis_name: 0})
            last = arr.isel({axis_name: cur - 1})
            arr = xr.concat(
                [first.expand_dims({axis_name: [0]}), arr, last.expand_dims({axis_name: [0]})],
                dim=axis_name,
            )
            return arr.isel({axis_name: slice(0, target_len)})

        if cur == target_len + 1:
            # drop one at each side
            return arr.isel({axis_name: slice(1, cur - 1)})

        # If larger, center-crop
        if cur > target_len:
            off = (cur - target_len) // 2
            return arr.isel({axis_name: slice(off, off + target_len)})

        # If smaller, edge-pad
        need = target_len - cur
        left_n = need // 2
        right_n = need - left_n

        if cur == 0:
            raise ValueError(f"Cannot pad axis '{axis_name}' of length 0 to {target_len}.")

        left_block = xr.concat([arr.isel({axis_name: 0})] * left_n, dim=axis_name) if left_n else None
        right_block = xr.concat([arr.isel({axis_name: cur - 1})] * right_n, dim=axis_name) if right_n else None

        parts = []
        if left_block is not None:
            parts.append(left_block)
        parts.append(arr)
        if right_block is not None:
            parts.append(right_block)

        return xr.concat(parts, dim=axis_name)

    def _harmonize_to_ref_shape(slices: list[xr.DataArray]) -> list[xr.DataArray]:
        """Force all slices to match the first slice's (y,x) sizes."""
        ref_y, ref_x = slices[0].sizes["y"], slices[0].sizes["x"]
        out = []
        for s in slices:
            s2 = _match_axis_len(s, ref_x, "x")
            s2 = _match_axis_len(s2, ref_y, "y")
            out.append(s2)
        return out

    def _stack_one_var(files: list[str], *, var_name: str) -> xr.DataArray | None:
        """
        Read a sequence of rasters for one variable, stack along dim_name.
        Coordinates for dim_name are integer steps parsed from filenames.
        """
        triplets: list[tuple[int, str, xr.DataArray]] = []

        for f in files:
            if not os.path.exists(f):
                if skip_missing:
                    continue
                raise FileNotFoundError(f)

            da = _open_ascii(f)
            da.name = var_name

            if crs:
                da.rio.write_crs(crs, inplace=True)

            nd = da.rio.nodata if nodata is None else nodata
            if nd is not None:
                da.rio.write_nodata(nd, inplace=True)

            step_val = _parse_step(f)
            triplets.append((step_val, f, da))

        if not triplets:
            return None

        triplets.sort(key=lambda t: t[0])
        steps = [t[0] for t in triplets]
        arrays = [t[2] for t in triplets]

        if len(arrays) > 1:
            if strict_align:
                y0, x0 = arrays[0].sizes["y"], arrays[0].sizes["x"]
                need_harmonize = any(a.sizes["y"] != y0 or a.sizes["x"] != x0 for a in arrays[1:])
                if need_harmonize:
                    arrays = _harmonize_to_ref_shape(arrays)
            else:
                arrays = _harmonize_to_ref_shape(arrays)

        stack = xr.concat(arrays, dim=dim_name).assign_coords({dim_name: (dim_name, steps)})
        stack.name = var_name
        return stack

    def _center_x(da: xr.DataArray) -> xr.DataArray:
        """Average adjacent x faces to cell-centres (requires x>=2)."""
        return 0.5 * (da.isel(x=slice(0, -1)) + da.isel(x=slice(1, None)))

    def _center_y(da: xr.DataArray) -> xr.DataArray:
        """Average adjacent y faces to cell-centres (requires y>=2)."""
        return 0.5 * (da.isel(y=slice(0, -1)) + da.isel(y=slice(1, None)))

    # -------------------- main workflow --------------------

    if var_map is None:
        var_map = {
            "water_depth": "wd",
            "vel_x": "Vx",
            "vel_y": "Vy",
            "flux_x": "Qx",
            "flux_y": "Qy",
        }

    raw: dict[str, xr.DataArray] = {}
    for vname, ext in var_map.items():
        files = _build_range(ext)
        da = _stack_one_var(files, var_name=vname)
        if da is not None:
            raw[vname] = da

    if "water_depth" not in raw:
        raise ValueError("Could not read 'water_depth' series—can't define the target grid.")

    # Ensure canonical order
    wd = raw["water_depth"].transpose(dim_name, "y", "x")

    # ---- build REFERENCE_TS datetime coordinate ----
    steps = wd[dim_name].values.astype("int64")
    start64 = np.datetime64(reference_start, "ns")
    dt = np.timedelta64(int(dt_minutes), "m")
    ref_ts = start64 + (steps - int(start)) * dt

    # Target grid comes from wd
    x_wd = wd["x"].values
    y_wd = wd["y"].values
    nx_wd = x_wd.size
    ny_wd = y_wd.size

    def _align_to_wd_grid(da: xr.DataArray) -> xr.DataArray:
        """Force array to match wd (y,x) sizes, and assign wd coords."""
        da = _match_axis_len(da, nx_wd, "x")
        da = _match_axis_len(da, ny_wd, "y")
        return da.assign_coords(x=x_wd, y=y_wd)

    # Dataset with datetime time coord
    ds = xr.Dataset(coords={dim_name: ref_ts, "y": y_wd, "x": x_wd})
    ds[dim_name].attrs.update(standard_name="time", long_name="reference timestamp")

    # Always write water depth
    ds["water_depth"] = wd.assign_coords({dim_name: ref_ts})
    ds["water_depth"].attrs.update(long_name="water depth", units="m")

    # ---- Cell-centred face fields (safe) ----
    if "vel_x" in raw:
        Vx = raw["vel_x"].transpose(dim_name, "y", "x")
        if Vx.sizes.get("x", 0) < 2:
            print(f"WARNING: vel_x has x<2 (sizes={dict(Vx.sizes)}). Skipping vel_x_c.")
        else:
            vx_c = _align_to_wd_grid(_center_x(Vx))
            ds["vel_x_c"] = vx_c.assign_coords({dim_name: ref_ts})
            ds["vel_x_c"].attrs.update(long_name="cell-centred velocity x", units="m s-1")

    if "vel_y" in raw:
        Vy = raw["vel_y"].transpose(dim_name, "y", "x")
        if Vy.sizes.get("y", 0) < 2:
            print(f"WARNING: vel_y has y<2 (sizes={dict(Vy.sizes)}). Skipping vel_y_c.")
        else:
            vy_c = _align_to_wd_grid(_center_y(Vy))
            ds["vel_y_c"] = vy_c.assign_coords({dim_name: ref_ts})
            ds["vel_y_c"].attrs.update(long_name="cell-centred velocity y", units="m s-1")

    if "flux_x" in raw:
        Qx = raw["flux_x"].transpose(dim_name, "y", "x")
        if Qx.sizes.get("x", 0) < 2:
            print(f"WARNING: flux_x has x<2 (sizes={dict(Qx.sizes)}). Skipping flux_x_c.")
        else:
            qx_c = _align_to_wd_grid(_center_x(Qx))
            ds["flux_x_c"] = qx_c.assign_coords({dim_name: ref_ts})
            ds["flux_x_c"].attrs.update(long_name="cell-centred discharge x", units="m3 s-1")

    if "flux_y" in raw:
        Qy = raw["flux_y"].transpose(dim_name, "y", "x")
        if Qy.sizes.get("y", 0) < 2:
            print(f"WARNING: flux_y has y<2 (sizes={dict(Qy.sizes)}). Skipping flux_y_c.")
        else:
            qy_c = _align_to_wd_grid(_center_y(Qy))
            ds["flux_y_c"] = qy_c.assign_coords({dim_name: ref_ts})
            ds["flux_y_c"].attrs.update(long_name="cell-centred discharge y", units="m3 s-1")

    # ---- Magnitudes ----
    if ("vel_x_c" in ds) and ("vel_y_c" in ds):
        ds["vel_mag"] = xr.apply_ufunc(np.hypot, ds["vel_x_c"], ds["vel_y_c"])
        ds["vel_mag"].attrs.update(long_name="speed magnitude", units="m s-1")

    if ("flux_x_c" in ds) and ("flux_y_c" in ds):
        ds["flux_mag"] = xr.apply_ufunc(np.hypot, ds["flux_x_c"], ds["flux_y_c"])
        ds["flux_mag"].attrs.update(long_name="discharge magnitude", units="m3 s-1")

    # ---- Representative per-cell fields (SAFE: no empty slicing) ----
    # discharge_cell: average of |Qx| and |Qy| after independent cell-centring
    if ("flux_x" in raw) and ("flux_y" in raw):
        Qx = raw["flux_x"].transpose(dim_name, "y", "x")
        Qy = raw["flux_y"].transpose(dim_name, "y", "x")

        if (Qx.sizes.get("x", 0) < 2) or (Qy.sizes.get("y", 0) < 2):
            print(
                "WARNING: skipping discharge_cell (face grid too small). "
                f"Qx sizes={dict(Qx.sizes)}, Qy sizes={dict(Qy.sizes)}"
            )
        else:
            Qx_c = _align_to_wd_grid(_center_x(np.abs(Qx)))
            Qy_c = _align_to_wd_grid(_center_y(np.abs(Qy)))
            Qrep = 0.5 * (Qx_c + Qy_c)
            ds["discharge_cell"] = Qrep.assign_coords({dim_name: ref_ts})
            ds["discharge_cell"].attrs.update(
                long_name="representative per-cell discharge (avg |faces|)",
                units="m3 s-1",
            )

    # speed_cell: average of |Vx| and |Vy| after independent cell-centring
    if ("vel_x" in raw) and ("vel_y" in raw):
        Vx = raw["vel_x"].transpose(dim_name, "y", "x")
        Vy = raw["vel_y"].transpose(dim_name, "y", "x")

        if (Vx.sizes.get("x", 0) < 2) or (Vy.sizes.get("y", 0) < 2):
            print(
                "WARNING: skipping speed_cell (face grid too small). "
                f"Vx sizes={dict(Vx.sizes)}, Vy sizes={dict(Vy.sizes)}"
            )
        else:
            Vx_c = _align_to_wd_grid(_center_x(np.abs(Vx)))
            Vy_c = _align_to_wd_grid(_center_y(np.abs(Vy)))
            Srep = 0.5 * (Vx_c + Vy_c)
            ds["speed_cell"] = Srep.assign_coords({dim_name: ref_ts})
            ds["speed_cell"].attrs.update(
                long_name="representative per-cell speed (avg |faces|)",
                units="m s-1",
            )

    # ---- Wet mask ----
    if wet_threshold > 0:
        wet = ds["water_depth"] >= float(wet_threshold)
        for vn in [
            "vel_x_c", "vel_y_c", "vel_mag",
            "flux_x_c", "flux_y_c", "flux_mag",
            "discharge_cell", "speed_cell",
        ]:
            if vn in ds:
                ds[vn] = ds[vn].where(wet)

    # ---- CRS / nodata ----
    for v in ds.data_vars:
        if crs:
            ds[v].rio.write_crs(crs, inplace=True)
        nd = (ds[v].rio.nodata if nodata is None else nodata)
        if nd is not None:
            ds[v].rio.write_nodata(nd, inplace=True)
            if use_nan_fill:
                ds[v] = ds[v].where(ds[v] != nd)

    # ---- Optional realization dim ----
    if realization is not None:
        ds = ds.expand_dims({"realization": [int(realization)]})
        ds = ds.transpose("realization", dim_name, "y", "x", ...)

    # Remove fill attributes that can conflict with encoding
    for v in ds.data_vars:
        ds[v].attrs.pop("_FillValue", None)
        ds[v].attrs.pop("missing_value", None)

    # ---- Encoding ----
    for v in ds.data_vars:
        enc = dict(zlib=True, complevel=int(complevel), dtype=dtype)
        nd = (ds[v].rio.nodata if nodata is None else nodata)
        if (nd is not None) and (not use_nan_fill):
            enc["_FillValue"] = float(nd)
        ds[v].encoding = enc

    ds.attrs.update(
        Conventions="CF-1.8",
        source="LISFLOOD single-member ASCII series, cell-centred on water_depth grid",
        deterministic="true" if realization is None else f"realization {int(realization)}",
    )

    ds.to_netcdf(out_nc)
    return out_nc

####################################################################################################
#########################ASCII FILE SUBGRID TO NETCFILE BECAUSE IT HAS ONLY ONE ####################
#########################VARIABLE ##################################################################

import os
import numpy as np
import xarray as xr
import rioxarray as rxr


def ascii_wdfp_subgrid_to_netcdf(
    *,
    out_nc: str,
    folder: str,
    base: str,
    start: int,
    end: int,
    width: int = 4,
    ext: str = "wdfp",                  # <-- your termination
    var_name: str = "water_depth",      # name inside NetCDF
    crs: str = "EPSG:2056",
    nodata: float | None = None,
    dtype: str = "float32",
    complevel: int = 4,
    chunks: dict | None = None,
    strict_align: bool = True,
    use_nan_fill: bool = False,
    skip_missing: bool = False,

    # ---- TIME SETTINGS ----
    dim_name: str = "REFERENCE_TS",
    reference_start: str = "2022-05-05T12:00:00.000000000",
    dt_minutes: int = 60,

    realization: int | None = None,
) -> str:
    """
    Convert LISFLOOD ASCII grids with only one variable terminated by `.wdfp`
    into a NetCDF with a datetime64[ns] time coordinate.

    Expected filenames:
      <folder>/<base>-0000.wdfp
      <folder>/<base>-0001.wdfp
      ...
    """

    def _open_ascii(path: str) -> xr.DataArray:
        try:
            da = rxr.open_rasterio(path, masked=True, chunks=chunks)
        except Exception:
            da = rxr.open_rasterio(path, masked=True, chunks=chunks, driver="AAIGrid")
        return da.squeeze("band", drop=True)  # (y,x)

    def _build_range() -> list[str]:
        return [os.path.join(folder, f"{base}-{i:0{width}d}.{ext}") for i in range(start, end + 1)]

    def _parse_step(fname: str) -> int:
        b = os.path.basename(fname)
        stem, _ = os.path.splitext(b)  # "<base>-0000"
        if "-" not in stem:
            raise ValueError(f"Cannot parse timestep (no '-') from filename: {b}")
        tail = stem.split("-")[-1]
        if not tail.isdigit():
            raise ValueError(f"Cannot parse timestep (not digits) from filename: {b}")
        return int(tail)

    def _match_axis_len(arr: xr.DataArray, target_len: int, axis_name: str) -> xr.DataArray:
        cur = arr.sizes.get(axis_name, None)
        if cur is None:
            raise KeyError(f"Axis '{axis_name}' not found in array dims {arr.dims}")
        if cur == target_len:
            return arr

        if cur == target_len - 1:
            first = arr.isel({axis_name: 0})
            last = arr.isel({axis_name: cur - 1})
            arr = xr.concat(
                [first.expand_dims({axis_name: [0]}), arr, last.expand_dims({axis_name: [0]})],
                dim=axis_name,
            )
            return arr.isel({axis_name: slice(0, target_len)})

        if cur == target_len + 1:
            return arr.isel({axis_name: slice(1, cur - 1)})

        if cur > target_len:
            off = (cur - target_len) // 2
            return arr.isel({axis_name: slice(off, off + target_len)})

        # pad if smaller
        need = target_len - cur
        if cur == 0:
            raise ValueError(f"Cannot pad axis '{axis_name}' of length 0 to {target_len}.")
        left_n = need // 2
        right_n = need - left_n
        left_block = xr.concat([arr.isel({axis_name: 0})] * left_n, dim=axis_name) if left_n else None
        right_block = xr.concat([arr.isel({axis_name: cur - 1})] * right_n, dim=axis_name) if right_n else None

        parts = []
        if left_block is not None:
            parts.append(left_block)
        parts.append(arr)
        if right_block is not None:
            parts.append(right_block)
        return xr.concat(parts, dim=axis_name)

    # ---- read & stack ----
    files = _build_range()

    triplets: list[tuple[int, xr.DataArray]] = []
    for f in files:
        if not os.path.exists(f):
            if skip_missing:
                continue
            raise FileNotFoundError(f)

        da = _open_ascii(f)
        if crs:
            da.rio.write_crs(crs, inplace=True)

        nd = da.rio.nodata if nodata is None else nodata
        if nd is not None:
            da.rio.write_nodata(nd, inplace=True)

        step = _parse_step(f)
        triplets.append((step, da))

    if not triplets:
        raise ValueError("No input .wdfp files found to write.")

    triplets.sort(key=lambda t: t[0])
    steps = np.array([t[0] for t in triplets], dtype="int64")
    arrays = [t[1] for t in triplets]

    # harmonize shapes if needed
    if len(arrays) > 1:
        y0, x0 = arrays[0].sizes["y"], arrays[0].sizes["x"]
        need_harmonize = any(a.sizes["y"] != y0 or a.sizes["x"] != x0 for a in arrays[1:])
        if need_harmonize or (not strict_align):
            arrays2 = []
            for a in arrays:
                a2 = _match_axis_len(a, x0, "x")
                a2 = _match_axis_len(a2, y0, "y")
                arrays2.append(a2)
            arrays = arrays2

    stack = xr.concat(arrays, dim=dim_name).assign_coords({dim_name: (dim_name, steps)})
    stack.name = var_name
    stack = stack.transpose(dim_name, "y", "x")

    # ---- time coordinate ----
    start64 = np.datetime64(reference_start, "ns")
    dt = np.timedelta64(int(dt_minutes), "m")
    ref_ts = start64 + (steps - int(start)) * dt

    # ---- dataset ----
    x = stack["x"].values
    y = stack["y"].values

    ds = xr.Dataset(coords={dim_name: ref_ts, "y": y, "x": x})
    ds[dim_name].attrs.update(standard_name="time", long_name="reference timestamp")

    ds[var_name] = stack.assign_coords({dim_name: ref_ts})
    ds[var_name].attrs.update(long_name="water depth", units="m")

    # CRS / nodata / nan fill
    if crs:
        ds[var_name].rio.write_crs(crs, inplace=True)

    nd = ds[var_name].rio.nodata if nodata is None else nodata
    if nd is not None:
        ds[var_name].rio.write_nodata(nd, inplace=True)
        if use_nan_fill:
            ds[var_name] = ds[var_name].where(ds[var_name] != nd)

    # Optional realization dim
    if realization is not None:
        ds = ds.expand_dims({"realization": [int(realization)]})
        ds = ds.transpose("realization", dim_name, "y", "x")

    # Remove fill attrs that can conflict with encoding
    ds[var_name].attrs.pop("_FillValue", None)
    ds[var_name].attrs.pop("missing_value", None)

    # Encoding
    enc = dict(zlib=True, complevel=int(complevel), dtype=dtype)
    if (nd is not None) and (not use_nan_fill):
        enc["_FillValue"] = float(nd)
    ds[var_name].encoding = enc

    ds.attrs.update(
        Conventions="CF-1.8",
        source="LISFLOOD single-variable ASCII series (.wdfp)",
        deterministic="true" if realization is None else f"realization {int(realization)}",
    )

    ds.to_netcdf(out_nc)
    return out_nc

################################################################################################
##########################FORECAST DATA SIMULATED TO NETCDFILE##################################
import os
import re
import numpy as np
import rasterio
from netCDF4 import Dataset


def lisflood_ensemble_to_forecastlike_netcdf(
    *,
    out_nc: str,
    member_folders: list[str],
    base: str,
    start: int,
    end: int,
    width: int = 4,

    var_map: dict[str, str] | None = None,
    crs: str = "EPSG:2056",
    nodata: float | None = None,
    dtype_data: str = "float32",
    complevel: int = 4,
    chunk_xy: tuple[int, int] | None = (256, 256),
    step_regex: str = r"-(\d{4})\.(?:\w+)$",
    strict_align: bool = True,
    skip_missing: bool = False,

    reference_start: str = "2022-05-05T12:00:00.000000000",
    dt_minutes: int = 60,

    dim_forecast_ref: str = "forecast_reference_time",
    dim_lead: str = "lead_time",
    dim_realization: str = "realization",
) -> str:
    """
    Write COSMO-like NetCDF in a streaming way (low memory):
      dims: (forecast_reference_time=1, lead_time=N, realization=M, y, x)

    Notes:
      - Adds realization as a proper coordinate variable
      - Does NOT create a 'crs' data variable (CRS saved as global attr only)
    """

    if var_map is None:
        var_map = {
            "water_depth": "wd",
            "vel_x": "Vx",
            "vel_y": "Vy",
            "flux_x": "Qx",
            "flux_y": "Qy",
        }

    # ---- helpers ----
    def _build_path(folder: str, ext: str, i: int) -> str:
        return os.path.join(folder, f"{base}-{i:0{width}d}.{ext}")

    def _read_grid(path: str):
        with rasterio.open(path) as src:
            a = src.read(1, masked=False)
            tx = src.transform
            fn_nd = src.nodata
            h, w = a.shape

            # cell centers
            x0 = tx.c + (0.5 * tx.a)
            y0 = tx.f + (0.5 * tx.e)
            x = x0 + np.arange(w, dtype=np.float64) * tx.a
            y = y0 + np.arange(h, dtype=np.float64) * tx.e
            return a.astype(np.float32, copy=False), x, y, tx, fn_nd

    def _center_x(arr):
        return 0.5 * (arr[:, :-1] + arr[:, 1:])

    def _center_y(arr):
        return 0.5 * (arr[:-1, :] + arr[1:, :])

    def _match_axis_len_np(arr: np.ndarray, target_len: int, axis: int) -> np.ndarray:
        cur = arr.shape[axis]
        if cur == target_len:
            return arr
        if cur == target_len - 1:
            if axis == 1:
                left = arr[:, :1]
                right = arr[:, -1:]
                arr2 = np.concatenate([left, arr, right], axis=1)
                return arr2[:, :target_len]
            else:
                top = arr[:1, :]
                bot = arr[-1:, :]
                arr2 = np.concatenate([top, arr, bot], axis=0)
                return arr2[:target_len, :]
        if cur == target_len + 1:
            if axis == 1:
                return arr[:, 1:-1]
            else:
                return arr[1:-1, :]
        if cur > target_len:
            off = (cur - target_len) // 2
            if axis == 1:
                return arr[:, off:off + target_len]
            else:
                return arr[off:off + target_len, :]
        need = target_len - cur
        left_n = need // 2
        right_n = need - left_n
        if axis == 1:
            left = np.repeat(arr[:, :1], left_n, axis=1)
            right = np.repeat(arr[:, -1:], right_n, axis=1)
            return np.concatenate([left, arr, right], axis=1)
        else:
            top = np.repeat(arr[:1, :], left_n, axis=0)
            bot = np.repeat(arr[-1:, :], right_n, axis=0)
            return np.concatenate([top, arr, bot], axis=0)

    # ---- determine step list and lead times ----
    step_ids = np.arange(start, end + 1, dtype=np.int64)
    n_lead = step_ids.size
    lead_seconds = (step_ids - int(start)) * float(dt_minutes) * 60.0  # float64

    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    init_dt64 = np.datetime64(reference_start, "ns")
    init_seconds = float((init_dt64 - epoch) / np.timedelta64(1, "s"))
    time_seconds = init_seconds + lead_seconds
    time_2d = time_seconds.reshape(1, n_lead).astype(np.float64)

    # ---- open first water_depth to define grid ----
    if not member_folders:
        raise ValueError("member_folders is empty")

    first_wd = _build_path(member_folders[0], var_map["water_depth"], int(step_ids[0]))
    if not os.path.exists(first_wd):
        raise FileNotFoundError(f"Cannot find first water_depth file: {first_wd}")

    wd0, x, y, tx_ref, nd_file = _read_grid(first_wd)
    ny, nx = wd0.shape

    nd = nd_file if nodata is None else nodata
    if nd is None:
        nd = np.float32(-9999.0)

    M = len(member_folders)

    # ---- create netcdf file ----
    os.makedirs(os.path.dirname(out_nc) or ".", exist_ok=True)
    with Dataset(out_nc, "w", format="NETCDF4") as nc:
        # dims
        nc.createDimension(dim_forecast_ref, 1)
        nc.createDimension(dim_lead, n_lead)
        nc.createDimension(dim_realization, M)
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)

        # global attrs (store CRS here; NOT as a data variable)
        nc.Conventions = "CF-1.8"
        nc.source = "LISFLOOD ASCII ensemble stacked to COSMO-like forecast format (streaming)"
        nc.setncattr("crs", crs)

        # coords
        v_fr = nc.createVariable(dim_forecast_ref, "f8", (dim_forecast_ref,))
        v_fr[:] = np.array([init_seconds], dtype=np.float64)
        v_fr.units = "seconds since 1970-01-01"
        v_fr.calendar = "proleptic_gregorian"
        v_fr.long_name = "forecast reference time"

        v_lt = nc.createVariable(dim_lead, "f8", (dim_lead,))
        v_lt[:] = lead_seconds.astype(np.float64)
        v_lt.units = "s"
        v_lt.long_name = "lead time"

        # IMPORTANT: realization coordinate variable (so it appears properly in xarray)
        v_rlz = nc.createVariable(dim_realization, "i4", (dim_realization,))
        v_rlz[:] = np.arange(M, dtype=np.int32)
        v_rlz.long_name = "ensemble member"

        v_x = nc.createVariable("x", "f8", ("x",))
        v_x[:] = x
        v_x.long_name = "x coordinate of projection"
        v_x.units = "m"

        v_y = nc.createVariable("y", "f8", ("y",))
        v_y[:] = y
        v_y.long_name = "y coordinate of projection"
        v_y.units = "m"

        v_time = nc.createVariable("time", "f8", (dim_forecast_ref, dim_lead))
        v_time[:, :] = time_2d
        v_time.units = "seconds since 1970-01-01"
        v_time.calendar = "proleptic_gregorian"
        v_time.long_name = "valid time"

        # chunking
        if chunk_xy is None:
            chunksizes = (1, 1, 1, ny, nx)
        else:
            cy, cx = chunk_xy
            chunksizes = (1, 1, 1, min(cy, ny), min(cx, nx))

        nc_dtype = "f4" if dtype_data == "float32" else "f8"

        def _make_var(name, long_name, units):
            v = nc.createVariable(
                name,
                nc_dtype,
                (dim_forecast_ref, dim_lead, dim_realization, "y", "x"),
                zlib=True,
                complevel=int(complevel),
                fill_value=np.float32(nd) if nc_dtype == "f4" else float(nd),
                chunksizes=chunksizes,
            )
            v.long_name = long_name
            v.units = units
            return v

        # create variables
        v_wd = _make_var("water_depth", "water depth", "m")

        v_vx = v_vy = v_qx = v_qy = None
        v_vmag = v_qmag = None
        if "vel_x" in var_map:
            v_vx = _make_var("vel_x_c", "cell-centred velocity x", "m s-1")
        if "vel_y" in var_map:
            v_vy = _make_var("vel_y_c", "cell-centred velocity y", "m s-1")
        if "flux_x" in var_map:
            v_qx = _make_var("flux_x_c", "cell-centred discharge x", "m3 s-1")
        if "flux_y" in var_map:
            v_qy = _make_var("flux_y_c", "cell-centred discharge y", "m3 s-1")
        if v_vx is not None and v_vy is not None:
            v_vmag = _make_var("vel_mag", "cell-centred speed (|vector|)", "m s-1")
        if v_qx is not None and v_qy is not None:
            v_qmag = _make_var("flux_mag", "cell-centred discharge magnitude (|vector|)", "m3 s-1")

        # ---- streaming write loops ----
        for r, folder in enumerate(member_folders):
            for ti, step in enumerate(step_ids):
                # water depth
                f_wd = _build_path(folder, var_map["water_depth"], int(step))
                if not os.path.exists(f_wd):
                    if skip_missing:
                        continue
                    raise FileNotFoundError(f_wd)

                a_wd, _, _, tx2, nd2 = _read_grid(f_wd)

                if strict_align:
                    if a_wd.shape != (ny, nx):
                        raise ValueError(f"[water_depth] shape mismatch: {f_wd} {a_wd.shape} vs {(ny, nx)}")
                    if tx2 != tx_ref:
                        raise ValueError(f"[water_depth] geotransform mismatch: {f_wd}")

                nd_here = nd2 if nodata is None else nodata
                if nd_here is not None:
                    a_wd = np.where(a_wd == nd_here, nd, a_wd).astype(np.float32, copy=False)

                v_wd[0, ti, r, :, :] = a_wd

                vx_c = vy_c = qx_c = qy_c = None

                # vel_x centered
                if v_vx is not None:
                    f = _build_path(folder, var_map["vel_x"], int(step))
                    if os.path.exists(f):
                        a, _, _, txv, ndv = _read_grid(f)
                        if strict_align and txv != tx_ref:
                            raise ValueError(f"[vel_x] geotransform mismatch: {f}")
                        a = np.where(a == (ndv if nodata is None else nodata), nd, a).astype(np.float32, copy=False)
                        vx_c = _center_x(a)
                        vx_c = _match_axis_len_np(vx_c, nx, axis=1)
                        vx_c = _match_axis_len_np(vx_c, ny, axis=0)
                        v_vx[0, ti, r, :, :] = vx_c
                    elif not skip_missing:
                        raise FileNotFoundError(f)

                # vel_y centered
                if v_vy is not None:
                    f = _build_path(folder, var_map["vel_y"], int(step))
                    if os.path.exists(f):
                        a, _, _, txv, ndv = _read_grid(f)
                        if strict_align and txv != tx_ref:
                            raise ValueError(f"[vel_y] geotransform mismatch: {f}")
                        a = np.where(a == (ndv if nodata is None else nodata), nd, a).astype(np.float32, copy=False)
                        vy_c = _center_y(a)
                        vy_c = _match_axis_len_np(vy_c, nx, axis=1)
                        vy_c = _match_axis_len_np(vy_c, ny, axis=0)
                        v_vy[0, ti, r, :, :] = vy_c
                    elif not skip_missing:
                        raise FileNotFoundError(f)

                # flux_x centered
                if v_qx is not None:
                    f = _build_path(folder, var_map["flux_x"], int(step))
                    if os.path.exists(f):
                        a, _, _, txv, ndv = _read_grid(f)
                        if strict_align and txv != tx_ref:
                            raise ValueError(f"[flux_x] geotransform mismatch: {f}")
                        a = np.where(a == (ndv if nodata is None else nodata), nd, a).astype(np.float32, copy=False)
                        qx_c = _center_x(a)
                        qx_c = _match_axis_len_np(qx_c, nx, axis=1)
                        qx_c = _match_axis_len_np(qx_c, ny, axis=0)
                        v_qx[0, ti, r, :, :] = qx_c
                    elif not skip_missing:
                        raise FileNotFoundError(f)

                # flux_y centered
                if v_qy is not None:
                    f = _build_path(folder, var_map["flux_y"], int(step))
                    if os.path.exists(f):
                        a, _, _, txv, ndv = _read_grid(f)
                        if strict_align and txv != tx_ref:
                            raise ValueError(f"[flux_y] geotransform mismatch: {f}")
                        a = np.where(a == (ndv if nodata is None else nodata), nd, a).astype(np.float32, copy=False)
                        qy_c = _center_y(a)
                        qy_c = _match_axis_len_np(qy_c, nx, axis=1)
                        qy_c = _match_axis_len_np(qy_c, ny, axis=0)
                        v_qy[0, ti, r, :, :] = qy_c
                    elif not skip_missing:
                        raise FileNotFoundError(f)

                # magnitudes
                if v_vmag is not None and (vx_c is not None) and (vy_c is not None):
                    v_vmag[0, ti, r, :, :] = np.hypot(vx_c, vy_c).astype(np.float32, copy=False)

                if v_qmag is not None and (qx_c is not None) and (qy_c is not None):
                    v_qmag[0, ti, r, :, :] = np.hypot(qx_c, qy_c).astype(np.float32, copy=False)

        nc.sync()

    return out_nc

   
#########################################################################################
########################################################################################

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_waterdepth_forecast_from_netcdfile(
    nc_path, out_dir,
    threshold=0.01,
    bg_pixel_size=1.0,
    bg_max_px=4096,
    layer="ch.swisstopo.swisstlm3d-karte-grau",
    case_label="Zell",
    extent=None,
    extent_units="auto",
    min_fig_height=6.0,
):
    # --- embedded Swisstopo WMS fetcher ---
    def _get_swisstopo_background_image_hq(xmin, xmax, ymin, ymax,
                                          pixel_size_m=1.0,
                                          layer="ch.swisstopo.swisstlm3d-karte-grau",
                                          max_px=4096):
        import requests
        from io import BytesIO

        width = int(max(1, round((xmax - xmin) / float(pixel_size_m))))
        height = int(max(1, round((ymax - ymin) / float(pixel_size_m))))
        scale = max(width / max_px, height / max_px, 1.0)
        width = int(width / scale)
        height = int(height / scale)

        params = {
            "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
            "LAYERS": layer,
            "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
            "CRS": "EPSG:2056",
            "WIDTH": width, "HEIGHT": height,
            "FORMAT": "image/png", "TRANSPARENT": "TRUE",
        }
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "image/png,image/*,*/*;q=0.8"}
        r = requests.get("https://wms.geo.admin.ch/", params=params, headers=headers, timeout=60)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))

    def _extent_to_meters(ext, units):
        if ext is None:
            return None
        xmin, xmax, ymin, ymax = ext
        if units == "m":
            return (xmin, xmax, ymin, ymax)
        if units == "km":
            return (xmin * 1000.0, xmax * 1000.0, ymin * 1000.0, ymax * 1000.0)
        # auto
        return (xmin * 1000.0, xmax * 1000.0, ymin * 1000.0, ymax * 1000.0) if max(abs(v) for v in ext) < 10000 else (xmin, xmax, ymin, ymax)

    def _fmt_dt(dt64):
        return np.datetime_as_string(dt64, unit="s")  # "YYYY-MM-DDTHH:MM:SS"

    os.makedirs(out_dir, exist_ok=True)
    ds = xr.open_dataset(nc_path)

    # --- required dims/vars ---
    if "forecast_reference_time" not in ds.dims:
        raise ValueError("Expected dim 'forecast_reference_time' in dataset.")
    if "lead_time" not in ds.dims:
        raise ValueError("Expected dim 'lead_time' in dataset.")
    if "realization" not in ds.dims:
        raise ValueError("Expected dim 'realization' in dataset.")
    if "water_depth" not in ds.data_vars:
        raise ValueError("Variable 'water_depth' not found in dataset.")

    # --- get init time from NetCDF ---
    # forecast_reference_time is stored as datetime64 in xarray view (nice!)
    frt = ds["forecast_reference_time"].values
    init_dt = frt[0] if np.ndim(frt) > 0 else frt
    init_txt = _fmt_dt(init_dt) if np.issubdtype(np.asarray(init_dt).dtype, np.datetime64) else str(init_dt)

    # --- domain extents ---
    x_all = ds["x"].values
    y_all = ds["y"].values
    dom_xmin, dom_xmax = float(np.min(x_all)), float(np.max(x_all))
    dom_ymin, dom_ymax = float(np.min(y_all)), float(np.max(y_all))

    if extent is None:
        xmin, xmax, ymin, ymax = dom_xmin, dom_xmax, dom_ymin, dom_ymax
    else:
        xmin_i, xmax_i, ymin_i, ymax_i = _extent_to_meters(extent, extent_units)
        xmin = max(dom_xmin, min(xmin_i, xmax_i))
        xmax = min(dom_xmax, max(xmin_i, xmax_i))
        ymin = max(dom_ymin, min(ymin_i, ymax_i))
        ymax = min(dom_ymax, max(ymin_i, ymax_i))

    plot_extent = (xmin, xmax, ymin, ymax)

    # selection slices
    xsel = slice(xmin, xmax) if x_all[0] < x_all[-1] else slice(xmax, xmin)
    ysel = slice(ymin, ymax) if y_all[0] < y_all[-1] else slice(ymax, ymin)

    # background
    try:
        bg = _get_swisstopo_background_image_hq(
            xmin, xmax, ymin, ymax,
            pixel_size_m=bg_pixel_size, layer=layer, max_px=bg_max_px
        )
    except Exception as e:
        print(f"⚠️  WMS fetch failed ({e}). Proceeding without background.")
        bg = None

    # ───────────────────────────────────────────────────────────────────
    # COLORS: same as deterministic (fixed bins + violet overflow)
    # IMPORTANT: len(colors) = len(edges)-1
    # ───────────────────────────────────────────────────────────────────
    edges = [0.05, 0.10, 0.30, 0.50, 1.0, 1.50, 2.0, 2.5, 3.0, 3.5]
    colors = [
        "#f7fcf0",  # 0.05–0.10
        "#ccebc5",  # 0.10–0.30
        "#a8ddb5",  # 0.30–0.50
        "#7bccc4",  # 0.50–1.00
        "#4eb3d3",  # 1.00–1.50
        "#2b8cbe",  # 1.50–2.00
        "#08589e",  # 2.00–2.50
        "#08306b",  # 2.50–3.00
        "#54278f",  # 3.00–3.50
    ]
    cmap_disc = ListedColormap(colors + ["#4d004b"])  # overflow >= 3.5
    cmap_disc.set_under((0, 0, 0, 0))
    norm = BoundaryNorm(edges, cmap_disc.N, extend="max")

    # figure sizing
    xspan = xmax - xmin
    yspan = ymax - ymin
    base_w = 8.0
    fig_h = max(min_fig_height, base_w * (yspan / xspan))

    # lead times are in seconds (float64) in your file
    lead_seconds = ds["lead_time"].values.astype(np.float64)
    lead_hours = lead_seconds / 3600.0

    reals = ds["realization"].values

    for r in reals:
        # keep the full COSMO dims: (forecast_reference_time, lead_time, realization, y, x)
        da_r = ds["water_depth"].sel(realization=r).sel(x=xsel, y=ysel)

        x_sel = da_r["x"].values
        y_sel = da_r["y"].values
        x_desc = x_sel[0] > x_sel[-1]
        y_desc = y_sel[0] > y_sel[-1]

        for t_idx in range(da_r.sizes["lead_time"]):
            # select one map (still has forecast_reference_time=1, drop by isel)
            da_map = da_r.isel(lead_time=t_idx, forecast_reference_time=0).transpose("y", "x")
            arr = da_map.values.astype(np.float32, copy=False)

            if y_desc:
                arr = arr[::-1, :]
            if x_desc:
                arr = arr[:, ::-1]

            arr = np.where(arr >= threshold, arr, np.nan)

            lh = float(lead_hours[t_idx])
            # title: forecast_reference_time + X hours
            if abs(lh - round(lh)) < 1e-6:
                lead_txt = f"+ {int(round(lh))} hours"
            else:
                lead_txt = f"+ {lh:.1f} hours"

            title = f"{case_label} — {init_txt} {lead_txt} (r={int(r)})"

            fig, ax = plt.subplots(figsize=(base_w, fig_h), dpi=150)

            if bg is not None:
                bg_np = np.array(bg)[::-1, :, :]
                ax.imshow(bg_np, extent=plot_extent, origin="lower", interpolation="nearest")

            ax.imshow(
                arr,
                extent=plot_extent,
                origin="lower",
                cmap=cmap_disc,
                norm=norm,
                interpolation="nearest"
            )

            ax.set_title(title, fontsize=20, fontweight="bold")
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_aspect("equal"); ax.grid(False)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.tick_params(left=False, right=False, bottom=False, top=False,
                           labelleft=False, labelright=False, labelbottom=False, labeltop=False)

            tag_zoom = "_zoom" if extent is not None else ""
            out_png = os.path.join(out_dir, f"Forecast_r{int(r)}_lead_time{t_idx}{tag_zoom}.png")
            plt.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ saved {out_png}")

    ds.close()

    ticks = [t for t in edges if (t >= threshold and t <= max(edges))] or [threshold, max(edges)]
    vmax = max(edges)
    return cmap_disc, norm, ticks, vmax


#####################################################################################################

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_water_depths_deterministic_from_netcdfile(
    nc_path,
    out_dir,
    threshold=0.01,
    vmax=3.5,
    bg_pixel_size=1.0,
    bg_max_px=4096,
    layer="ch.swisstopo.swisstlm3d-karte-grau",
    case_label="Zell",
    init_time_str=None,
    extent=None,
    extent_units="auto",
    min_fig_height=6.0,
):
    """
    Plot water depth-like variable from a NetCDF that has NO 'realization' dim.

    ✅ Now supports variable names in this priority:
        1) h
        2) water_depth
        3) wd

    Works with time dim named 'time' or 'REFERENCE_TS' (datetime64[ns]) or falls back to lead_time/step.
    Title uses the datetime from the NetCDF directly (e.g., 2022-05-05T12:00:00).
    """

    # ───────────────────────────────────────────────────────────────────
    # Swisstopo WMS helper (embedded)
    # ───────────────────────────────────────────────────────────────────
    def _get_swisstopo_background_image_hq(
        xmin, xmax, ymin, ymax,
        pixel_size_m=1.0,
        layer="ch.swisstopo.swisstlm3d-karte-grau",
        max_px=4096
    ):
        import requests
        from io import BytesIO

        width = int(max(1, round((xmax - xmin) / float(pixel_size_m))))
        height = int(max(1, round((ymax - ymin) / float(pixel_size_m))))
        scale = max(width / max_px, height / max_px, 1.0)
        width = int(width / scale)
        height = int(height / scale)

        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
            "CRS": "EPSG:2056",
            "WIDTH": width,
            "HEIGHT": height,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
        }
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "image/png,image/*,*/*;q=0.8"}
        r = requests.get("https://wms.geo.admin.ch/", params=params, headers=headers, timeout=60)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))

    # ───────────────────────────────────────────────────────────────────
    # small helpers
    # ───────────────────────────────────────────────────────────────────
    def _extent_to_meters(ext, units):
        if ext is None:
            return None
        xmin, xmax, ymin, ymax = ext
        if units == "m":
            return (xmin, xmax, ymin, ymax)
        if units == "km":
            return (xmin * 1000.0, xmax * 1000.0, ymin * 1000.0, ymax * 1000.0)
        # auto: treat small numbers as km, big as meters
        return (
            (xmin * 1000.0, xmax * 1000.0, ymin * 1000.0, ymax * 1000.0)
            if max(abs(v) for v in ext) < 10000
            else (xmin, xmax, ymin, ymax)
        )

    def _fmt_dt(dt64):
        return np.datetime_as_string(dt64, unit="s")  # "YYYY-MM-DDTHH:MM:SS"

    # ───────────────────────────────────────────────────────────────────
    # main
    # ───────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    ds = xr.open_dataset(nc_path)

    # ✅ Time dimension: prefer "time" (your NetCDF), else REFERENCE_TS, else lead_time/step
    if "time" in ds.dims:
        time_dim = "time"
    elif "REFERENCE_TS" in ds.dims:
        time_dim = "REFERENCE_TS"
    elif "lead_time" in ds.dims:
        time_dim = "lead_time"
    elif "step" in ds.dims:
        time_dim = "step"
    else:
        raise ValueError("Couldn't find a time dimension named 'time', 'REFERENCE_TS', 'lead_time', or 'step'.")

    # ✅ Variable name priority (prefer smoothed visualization field if available)
    if "h_smooth" in ds.data_vars:
        wd_name = "h_smooth"
    elif "h" in ds.data_vars:
        wd_name = "h"
    elif "water_depth" in ds.data_vars:
        wd_name = "water_depth"
    elif "wd" in ds.data_vars:
        wd_name = "wd"
    else:
        raise ValueError(
        "No variable found: expected one of "
        "['h_smooth', 'h', 'water_depth', 'wd']."
        )

    # ── domain extent
    x_all = ds["x"].values
    y_all = ds["y"].values
    dom_xmin, dom_xmax = float(np.min(x_all)), float(np.max(x_all))
    dom_ymin, dom_ymax = float(np.min(y_all)), float(np.max(y_all))

    if extent is None:
        xmin, xmax, ymin, ymax = dom_xmin, dom_xmax, dom_ymin, dom_ymax
    else:
        xmin_i, xmax_i, ymin_i, ymax_i = _extent_to_meters(extent, extent_units)
        xmin = max(dom_xmin, min(xmin_i, xmax_i))
        xmax = min(dom_xmax, max(xmin_i, xmax_i))
        ymin = max(dom_ymin, min(ymin_i, ymax_i))
        ymax = min(dom_ymax, max(ymin_i, ymax_i))

    plot_extent = (xmin, xmax, ymin, ymax)

    x_asc = x_all[0] < x_all[-1]
    y_asc = y_all[0] < y_all[-1]
    xsel = slice(xmin, xmax) if x_asc else slice(xmax, xmin)
    ysel = slice(ymin, ymax) if y_asc else slice(ymax, ymin)

    # background
    try:
        bg = _get_swisstopo_background_image_hq(
            xmin, xmax, ymin, ymax,
            pixel_size_m=bg_pixel_size,
            layer=layer,
            max_px=bg_max_px,
        )
    except Exception as e:
        print(f"⚠️  WMS fetch failed ({e}). Proceeding without background.")
        bg = None

    # ───────────────────────────────────────────────────────────────────
    # palette
    # ───────────────────────────────────────────────────────────────────
    edges = [0.01, 0.10, 0.30, 0.50, 1.0, 1.50, 2.0, 2.5, 3.0, 3.5]
    colors = [
        "#f7fcf0",
        "#ccebc5",
        "#a8ddb5",
        "#7bccc4",
        "#4eb3d3",
        "#2b8cbe",
        "#08589e",
        "#08306b",
        "#54278f",
    ]
    cmap_disc = ListedColormap(colors + ["#4d004b"])
    cmap_disc.set_under((0, 0, 0, 0))
    norm = BoundaryNorm(edges, cmap_disc.N, extend="max")

    # figure sizing
    xspan = xmax - xmin
    yspan = ymax - ymin
    base_w = 8.0
    fig_h = max(min_fig_height, base_w * (yspan / xspan))

    # subset data
    da_wd = ds[wd_name].sel(x=xsel, y=ysel)
    x_sel = da_wd["x"].values
    y_sel = da_wd["y"].values
    x_desc = x_sel[0] > x_sel[-1]
    y_desc = y_sel[0] > y_sel[-1]

    times = ds[time_dim].values

    for t_idx, t_val in enumerate(times):
        da_t = da_wd.isel({time_dim: t_idx}).transpose("y", "x")
        arr = da_t.values.astype(np.float32)

        if y_desc:
            arr = arr[::-1, :]
        if x_desc:
            arr = arr[:, ::-1]

        arr = np.where(arr >= threshold, arr, np.nan)

        if np.issubdtype(np.asarray(times).dtype, np.datetime64):
            t_txt = _fmt_dt(t_val)
            title = f"{case_label} — {wd_name} at {t_txt}"
            file_tag = t_txt.replace(":", "-")
        else:
            title = f"{case_label} — {wd_name} at step {t_val}"
            file_tag = str(t_val).replace(":", "-").replace(" ", "_")

        fig, ax = plt.subplots(figsize=(base_w, fig_h), dpi=150)

        if bg is not None:
            bg_np = np.array(bg)[::-1, :, :]
            ax.imshow(bg_np, extent=plot_extent, origin="lower", interpolation="bilinear")

        ax.imshow(
            arr,
            extent=plot_extent,
            origin="lower",
            cmap=cmap_disc,
            norm=norm,
            interpolation="bilinear",
        )

        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            left=False, right=False, bottom=False, top=False,
            labelleft=False, labelright=False, labelbottom=False, labeltop=False
        )

        tag_zoom = "_zoom" if extent is not None else ""
        out_png = os.path.join(out_dir, f"Combiprecip_{wd_name}_{file_tag}{tag_zoom}.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ saved {out_png}")

    ds.close()

    ticks = [t for t in edges if (t >= threshold and t <= vmax)] or [threshold, vmax]
    return cmap_disc, norm, ticks, vmax


###########################################################################################
####################### Plot hazard but with the 2 layers of Storme #####################
######################################################################################
import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm
from rasterio.features import rasterize
from rasterio.transform import Affine


def plot_water_depths_deterministic_from_netcdfile_storme(
    nc_path,
    out_dir,
    threshold=0.10,
    vmax=3.5,
    bg_pixel_size=1.0,
    bg_max_px=4096,
    layer="ch.swisstopo.swisstlm3d-karte-grau",
    case_label="Zell",
    init_time_str=None,
    extent=None,
    extent_units="auto",
    min_fig_height=6.0,
    storm_gpkg=None,
    layer_surface_runoff="prozessraum_wasser_oberflaechenabfluss_grundwasseraufstoss",
    layer_flooding="prozessraum_wasser_ueberschwemmung_uebermurung",
    color_surface_runoff="magenta",
    color_flooding="orange",
    alpha_surface_runoff=0.35,
    alpha_flooding=0.35,
    mask_gpkg="/storage/homefs/ge24z347/Zell_event/Data_forprocess/SWISSTLM3D_2025.gpkg",
    mask_layer="tlm_bb_bodenbedeckung",
    mask_objektart_col="objektart",
    mask_objektart_value="Stehende Gewaesser",
):
    """
    Plot deterministic water-depth maps from a NetCDF and overlay STORME polygons,
    excluding standing-water polygons from the raster and from the STORME polygons.

    Supported thresholds:
        0.01 or 0.10

    Final class is >= 3.00
    """

    def _get_swisstopo_background_image_hq(
        xmin, xmax, ymin, ymax,
        pixel_size_m=1.0,
        layer="ch.swisstopo.swisstlm3d-karte-grau",
        max_px=4096,
    ):
        import requests
        from io import BytesIO

        width = int(max(1, round((xmax - xmin) / float(pixel_size_m))))
        height = int(max(1, round((ymax - ymin) / float(pixel_size_m))))
        scale = max(width / max_px, height / max_px, 1.0)
        width = int(width / scale)
        height = int(height / scale)

        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
            "CRS": "EPSG:2056",
            "WIDTH": width,
            "HEIGHT": height,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/png,image/*,*/*;q=0.8",
        }

        r = requests.get(
            "https://wms.geo.admin.ch/",
            params=params,
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        return Image.open(BytesIO(r.content))

    def _extent_to_meters(ext, units):
        if ext is None:
            return None
        xmin, xmax, ymin, ymax = ext
        if units == "m":
            return (xmin, xmax, ymin, ymax)
        if units == "km":
            return (xmin * 1000.0, xmax * 1000.0, ymin * 1000.0, ymax * 1000.0)
        return (
            (xmin * 1000.0, xmax * 1000.0, ymin * 1000.0, ymax * 1000.0)
            if max(abs(v) for v in ext) < 10000
            else (xmin, xmax, ymin, ymax)
        )

    def _fmt_dt(dt64):
        return np.datetime_as_string(dt64, unit="s")

    def _load_and_clip_layer(gpkg_path, layer_name, xmin, xmax, ymin, ymax):
        if gpkg_path is None:
            return None
        try:
            gdf = gpd.read_file(gpkg_path, layer=layer_name)

            if gdf.empty:
                print(f"⚠️ Layer '{layer_name}' is empty in {gpkg_path}.")
                return None

            if gdf.crs is None:
                print(f"⚠️ Layer '{layer_name}' has no CRS. Assuming EPSG:2056.")
                gdf = gdf.set_crs(epsg=2056, allow_override=True)
            else:
                gdf = gdf.to_crs(epsg=2056)

            gdf = gdf[gdf.geometry.notnull()].copy()
            gdf = gdf[~gdf.geometry.is_empty].copy()

            try:
                gdf.geometry = gdf.geometry.make_valid()
            except Exception:
                pass

            gdf = gdf.cx[xmin:xmax, ymin:ymax].copy()

            if gdf.empty:
                print(f"ℹ️ No features from '{layer_name}' inside plot extent.")
                return None

            return gdf

        except Exception as e:
            print(f"⚠️ Could not load layer '{layer_name}' from '{gpkg_path}': {e}")
            return None

    def _subtract_mask(source_gdf, mask_gdf, name="layer"):
        if source_gdf is None or source_gdf.empty:
            return source_gdf
        if mask_gdf is None or mask_gdf.empty:
            return source_gdf

        try:
            source_gdf = source_gdf.copy()
            mask_gdf = mask_gdf.copy()

            try:
                source_gdf.geometry = source_gdf.geometry.make_valid()
                mask_gdf.geometry = mask_gdf.geometry.make_valid()
            except Exception:
                pass

            source_gdf = source_gdf[source_gdf.geometry.notnull()].copy()
            source_gdf = source_gdf[~source_gdf.geometry.is_empty].copy()
            mask_gdf = mask_gdf[mask_gdf.geometry.notnull()].copy()
            mask_gdf = mask_gdf[~mask_gdf.geometry.is_empty].copy()

            result = gpd.overlay(source_gdf, mask_gdf[["geometry"]], how="difference")

            if result.empty:
                print(f"ℹ️ After masking, '{name}' became empty.")
                return None

            result = result[result.geometry.notnull()].copy()
            result = result[~result.geometry.is_empty].copy()
            result = result.explode(index_parts=False).reset_index(drop=True)

            print(f"ℹ️ Applied standing-water mask to '{name}'.")
            return result

        except Exception as e:
            print(f"⚠️ Could not apply mask to '{name}': {e}")
            return source_gdf

    def _mask_raster_with_polygons(arr, x_coords, y_coords, mask_gdf):
        if mask_gdf is None or mask_gdf.empty:
            return arr

        try:
            geoms = [geom for geom in mask_gdf.geometry if geom is not None and not geom.is_empty]
            if not geoms:
                return arr

            x_coords = np.asarray(x_coords)
            y_coords = np.asarray(y_coords)

            nrows, ncols = arr.shape

            if len(x_coords) > 1:
                dx = abs(float(x_coords[1] - x_coords[0]))
            else:
                dx = 1.0

            if len(y_coords) > 1:
                dy = abs(float(y_coords[1] - y_coords[0]))
            else:
                dy = 1.0

            xmin_pix = float(np.min(x_coords)) - dx / 2.0
            ymax_pix = float(np.max(y_coords)) + dy / 2.0

            transform = Affine(dx, 0.0, xmin_pix, 0.0, -dy, ymax_pix)

            mask_int = rasterize(
                [(geom, 1) for geom in geoms],
                out_shape=(nrows, ncols),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype="uint8",
            )

            # Important: match origin="lower" used in imshow
            mask_int = np.flipud(mask_int)

            masked_pixels = int(mask_int.sum())
            print(f"ℹ️ Standing-water masked pixels: {masked_pixels}")

            arr_masked = arr.copy()
            arr_masked[mask_int == 1] = np.nan
            print("ℹ️ Applied standing-water mask to raster.")
            return arr_masked

        except Exception as e:
            print(f"⚠️ Could not apply standing-water mask to raster: {e}")
            return arr

    def _raster_extent_from_coords(x_coords, y_coords):
        x_coords = np.asarray(x_coords)
        y_coords = np.asarray(y_coords)

        if len(x_coords) > 1:
            dx = abs(float(x_coords[1] - x_coords[0]))
        else:
            dx = 1.0

        if len(y_coords) > 1:
            dy = abs(float(y_coords[1] - y_coords[0]))
        else:
            dy = 1.0

        xmin = float(np.min(x_coords)) - dx / 2.0
        xmax = float(np.max(x_coords)) + dx / 2.0
        ymin = float(np.min(y_coords)) - dy / 2.0
        ymax = float(np.max(y_coords)) + dy / 2.0
        return (xmin, xmax, ymin, ymax)

    os.makedirs(out_dir, exist_ok=True)
    ds = xr.open_dataset(nc_path)

    if "time" in ds.dims:
        time_dim = "time"
    elif "REFERENCE_TS" in ds.dims:
        time_dim = "REFERENCE_TS"
    elif "lead_time" in ds.dims:
        time_dim = "lead_time"
    elif "step" in ds.dims:
        time_dim = "step"
    else:
        raise ValueError(
            "Couldn't find a time dimension named 'time', 'REFERENCE_TS', 'lead_time', or 'step'."
        )

    if "h_smooth" in ds.data_vars:
        wd_name = "h_smooth"
    elif "h" in ds.data_vars:
        wd_name = "h"
    elif "water_depth" in ds.data_vars:
        wd_name = "water_depth"
    elif "wd" in ds.data_vars:
        wd_name = "wd"
    else:
        raise ValueError(
            "No variable found: expected one of ['h_smooth', 'h', 'water_depth', 'wd']."
        )

    x_all = ds["x"].values
    y_all = ds["y"].values
    dom_xmin, dom_xmax = float(np.min(x_all)), float(np.max(x_all))
    dom_ymin, dom_ymax = float(np.min(y_all)), float(np.max(y_all))

    if extent is None:
        xmin, xmax, ymin, ymax = dom_xmin, dom_xmax, dom_ymin, dom_ymax
    else:
        xmin_i, xmax_i, ymin_i, ymax_i = _extent_to_meters(extent, extent_units)
        xmin = max(dom_xmin, min(xmin_i, xmax_i))
        xmax = min(dom_xmax, max(xmin_i, xmax_i))
        ymin = max(dom_ymin, min(ymin_i, ymax_i))
        ymax = min(dom_ymax, max(ymin_i, ymax_i))

    plot_extent = (xmin, xmax, ymin, ymax)

    x_asc = x_all[0] < x_all[-1]
    y_asc = y_all[0] < y_all[-1]
    xsel = slice(xmin, xmax) if x_asc else slice(xmax, xmin)
    ysel = slice(ymin, ymax) if y_asc else slice(ymax, ymin)

    try:
        bg = _get_swisstopo_background_image_hq(
            xmin, xmax, ymin, ymax,
            pixel_size_m=bg_pixel_size,
            layer=layer,
            max_px=bg_max_px,
        )
    except Exception as e:
        print(f"⚠️ WMS fetch failed ({e}). Proceeding without background.")
        bg = None

    full_edges = [0.01, 0.10, 0.30, 0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 100.0]
    full_colors = [
        "#f7fcf0",
        "#ccebc5",
        "#a8ddb5",
        "#7bccc4",
        "#4eb3d3",
        "#2b8cbe",
        "#08589e",
        "#08306b",
        "#54278f",
    ]

    valid_thresholds = [0.01, 0.10]
    if threshold not in valid_thresholds:
        raise ValueError(f"threshold must be one of {valid_thresholds}, got {threshold}")

    start_idx = full_edges.index(threshold)
    edges = full_edges[start_idx:]
    colors = full_colors[start_idx:]

    cmap_disc = ListedColormap(colors)
    cmap_disc.set_under((0, 0, 0, 0))
    cmap_disc.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(edges, cmap_disc.N, clip=False)

    xspan = xmax - xmin
    yspan = ymax - ymin
    base_w = 8.0
    fig_h = max(min_fig_height, base_w * (yspan / xspan))

    da_wd = ds[wd_name].sel(x=xsel, y=ysel)
    x_sel = da_wd["x"].values
    y_sel = da_wd["y"].values
    x_desc = x_sel[0] > x_sel[-1]
    y_desc = y_sel[0] > y_sel[-1]

    raster_extent = _raster_extent_from_coords(x_sel, y_sel)

    print(
        "Raster subset bounds:",
        float(np.min(x_sel)), float(np.max(x_sel)),
        float(np.min(y_sel)), float(np.max(y_sel))
    )
    print("Raster image extent:", raster_extent)

    gdf_surface_runoff = _load_and_clip_layer(
        storm_gpkg, layer_surface_runoff, xmin, xmax, ymin, ymax
    )
    gdf_flooding = _load_and_clip_layer(
        storm_gpkg, layer_flooding, xmin, xmax, ymin, ymax
    )

    mask_gdf = _load_and_clip_layer(
        mask_gpkg, mask_layer, xmin, xmax, ymin, ymax
    )

    if mask_gdf is not None:
        if mask_objektart_col in mask_gdf.columns:
            print(
                "Mask unique objektart before filter:",
                mask_gdf[mask_objektart_col].astype(str).str.strip().unique()[:20]
            )

            mask_gdf = mask_gdf[
                mask_gdf[mask_objektart_col].astype(str).str.strip().str.lower()
                == mask_objektart_value.strip().lower()
            ].copy()

            if mask_gdf.empty:
                print(
                    f"ℹ️ No polygons found in '{mask_layer}' with "
                    f"{mask_objektart_col} = '{mask_objektart_value}'."
                )
                mask_gdf = None
            else:
                mask_gdf = mask_gdf[mask_gdf.geometry.notnull()].copy()
                mask_gdf = mask_gdf[~mask_gdf.geometry.is_empty].copy()
                mask_gdf = mask_gdf.explode(index_parts=False).reset_index(drop=True)

                try:
                    mask_gdf.geometry = mask_gdf.geometry.make_valid()
                except Exception:
                    pass

                print(
                    f"ℹ️ Loaded {len(mask_gdf)} standing-water polygon(s) "
                    f"from '{mask_gpkg}'."
                )
                print("Mask bounds:", mask_gdf.total_bounds)
        else:
            print(f"⚠️ Column '{mask_objektart_col}' not found in mask layer '{mask_layer}'.")
            mask_gdf = None

    gdf_surface_runoff = _subtract_mask(gdf_surface_runoff, mask_gdf, name="surface runoff")
    gdf_flooding = _subtract_mask(gdf_flooding, mask_gdf, name="flooding")

    times = ds[time_dim].values

    for t_idx, t_val in enumerate(times):
        da_t = da_wd.isel({time_dim: t_idx}).transpose("y", "x")
        arr = da_t.values.astype(np.float32)

        if y_desc:
            arr = arr[::-1, :]
        if x_desc:
            arr = arr[:, ::-1]

        arr = np.where(arr >= threshold, arr, np.nan)
        arr = _mask_raster_with_polygons(arr, x_sel, y_sel, mask_gdf)
        arr = np.ma.masked_invalid(arr)

        if np.issubdtype(np.asarray(times).dtype, np.datetime64):
            t_txt = _fmt_dt(t_val)
            title = f"{case_label} — {wd_name} at {t_txt}"
            file_tag = t_txt.replace(":", "-")
        else:
            title = f"{case_label} — {wd_name} at step {t_val}"
            file_tag = str(t_val).replace(":", "-").replace(" ", "_")

        fig, ax = plt.subplots(figsize=(base_w, fig_h), dpi=150)

        if bg is not None:
            bg_np = np.array(bg)[::-1, :, :]
            ax.imshow(
                bg_np,
                extent=plot_extent,
                origin="lower",
                interpolation="nearest",
            )

        ax.imshow(
            arr,
            extent=raster_extent,
            origin="lower",
            cmap=cmap_disc,
            norm=norm,
            interpolation="nearest",
        )

        if gdf_surface_runoff is not None and not gdf_surface_runoff.empty:
            gdf_surface_runoff.plot(
                ax=ax,
                facecolor=color_surface_runoff,
                edgecolor=color_surface_runoff,
                alpha=alpha_surface_runoff,
                zorder=20,
            )

        if gdf_flooding is not None and not gdf_flooding.empty:
            gdf_flooding.plot(
                ax=ax,
                facecolor=color_flooding,
                edgecolor=color_flooding,
                alpha=alpha_flooding,
                zorder=21,
            )

        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            left=False, right=False, bottom=False, top=False,
            labelleft=False, labelright=False, labelbottom=False, labeltop=False
        )

        tag_zoom = "_zoom" if extent is not None else ""
        out_png = os.path.join(
            out_dir,
            f"Combiprecip_{wd_name}_{file_tag}{tag_zoom}.png",
        )

        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        plt.savefig(
            out_png,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0
        )
        plt.close(fig)
        print(f"✅ saved {out_png}")

    ds.close()

    ticks = edges[:-1]
    return cmap_disc, norm, ticks, vmax

############################################################################################
################# Plot Combiprecip from non uniform grid accnugrid #######################
#########################################################################################
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image


def plot_h_unstructured_colored_edges(
    nc_path: str,
    out_dir: str,
    threshold: float = 0.01,
    vmax: float = 3.5,
    layer: str = "ch.swisstopo.swisstlm3d-karte-grau",
    bg_pixel_size: float = 1.0,
    bg_max_px: int = 4096,
    case_label: str = "Zell",
    init_time_str: str | None = None,
    extent=None,                 # (xmin, xmax, ymin, ymax) in EPSG:2056
    extent_units: str = "m",     # "m" or "km"
    linewidth: float = 0.12,
    dpi: int = 200,
    figsize=(10, 8),
):
    """
    Plot unstructured UGRID NetCDF variable h(time, mesh2d_face) as polygons,
    overlaid on swisstopo WMS background (EPSG:2056), with ParaView-like
    feature edges where edge color follows the same class color as faces.

    Expected variables/dims:
      - mesh2d_node_x(mesh2d_node), mesh2d_node_y(mesh2d_node)
      - mesh2d_face_nodes(mesh2d_face, max_n_face_nodes) padded with -1
      - h(time, mesh2d_face)

    Saves one PNG per timestep in out_dir.
    """

    def _extent_to_meters(ext, units):
        if ext is None:
            return None
        xmin, xmax, ymin, ymax = ext
        if units == "km":
            return xmin * 1000.0, xmax * 1000.0, ymin * 1000.0, ymax * 1000.0
        return xmin, xmax, ymin, ymax

    def _fmt_dt(dt64):
        return np.datetime_as_string(dt64, unit="s")

    def _get_swisstopo_background_image_hq(xmin, xmax, ymin, ymax, pixel_size_m=1.0, max_px=4096):
        import requests
        from io import BytesIO

        width = int(max(1, round((xmax - xmin) / float(pixel_size_m))))
        height = int(max(1, round((ymax - ymin) / float(pixel_size_m))))
        scale = max(width / max_px, height / max_px, 1.0)
        width = int(width / scale)
        height = int(height / scale)

        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
            "CRS": "EPSG:2056",
            "WIDTH": width,
            "HEIGHT": height,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
        }
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "image/png,image/*,*/*;q=0.8"}
        r = requests.get("https://wms.geo.admin.ch/", params=params, headers=headers, timeout=60)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))

    # -------------------------
    # class bins + colors (YOUR SETTINGS)
    # -------------------------
    class_edges = [threshold, 0.10, 0.30, 0.50, 1.0, 1.50, 2.0, 2.5, 3.0, vmax]
    class_colors = [
        "#f7fcf0",
        "#ccebc5",
        "#a8ddb5",
        "#7bccc4",
        "#4eb3d3",
        "#2b8cbe",
        "#08589e",
        "#08306b",
        "#54278f",
        "#4d004b",
    ]
    cmap_disc = ListedColormap(class_colors)
    norm = BoundaryNorm(class_edges, ncolors=cmap_disc.N, clip=False)

    # -------------------------
    # load dataset
    # -------------------------
    os.makedirs(out_dir, exist_ok=True)
    ds = xr.open_dataset(nc_path)

    if "h" not in ds:
        raise ValueError("Dataset has no variable 'h'.")

    if ("time" not in ds["h"].dims) or ("mesh2d_face" not in ds["h"].dims):
        raise ValueError(f"Expected h(time, mesh2d_face). Found: {ds['h'].dims}")

    x_node = ds["mesh2d_node_x"].values
    y_node = ds["mesh2d_node_y"].values
    faces = ds["mesh2d_face_nodes"].values
    times = ds["time"].values

    # -------------------------
    # extent (same behavior as your old script)
    # -------------------------
    dom_xmin, dom_xmax = float(np.nanmin(x_node)), float(np.nanmax(x_node))
    dom_ymin, dom_ymax = float(np.nanmin(y_node)), float(np.nanmax(y_node))

    if extent is None:
        xmin, xmax, ymin, ymax = dom_xmin, dom_xmax, dom_ymin, dom_ymax
    else:
        xmin_i, xmax_i, ymin_i, ymax_i = _extent_to_meters(extent, extent_units)

        # allow inverted order (like your example ymin>ymax)
        xmin_i, xmax_i = sorted([xmin_i, xmax_i])
        ymin_i, ymax_i = sorted([ymin_i, ymax_i])

        xmin = max(dom_xmin, xmin_i)
        xmax = min(dom_xmax, xmax_i)
        ymin = max(dom_ymin, ymin_i)
        ymax = min(dom_ymax, ymax_i)

    plot_extent = (xmin, xmax, ymin, ymax)

    # build polygons once
    polygons = []
    for face in faces:
        ids = face[face >= 0]
        polygons.append(np.column_stack([x_node[ids], y_node[ids]]))

    # background once
    try:
        bg = _get_swisstopo_background_image_hq(
            xmin, xmax, ymin, ymax,
            pixel_size_m=bg_pixel_size,
            max_px=bg_max_px
        )
    except Exception as e:
        print(f"⚠️ WMS fetch failed ({e}). Plotting without background.")
        bg = None

    # dummy mappable for colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_disc)
    sm.set_array([])

    init_tag = ""
    if init_time_str:
        init_tag = init_time_str.replace(":", "-")

    # -------------------------
    # plot each timestep
    # -------------------------
    for t_idx, t_val in enumerate(times):
        h = ds["h"].isel(time=t_idx).values.astype(np.float32)
        h_plot = np.where(h >= threshold, h, np.nan)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if bg is not None:
            bg_np = np.array(bg)[::-1, :, :]
            ax.imshow(bg_np, extent=plot_extent, origin="lower", interpolation="bilinear")

        # one collection per class so edgecolor == facecolor
        for i in range(len(class_edges) - 1):
            lo = class_edges[i]
            hi = class_edges[i + 1]
            col = class_colors[i]

            mask = np.isfinite(h_plot) & (h_plot >= lo) & (h_plot < hi)
            if not np.any(mask):
                continue

            idx = np.where(mask)[0]
            polys_i = [polygons[j] for j in idx]

            pc = PolyCollection(
                polys_i,
                facecolors=col,
                edgecolors=col,
                linewidths=linewidth,
                alpha=1.0,
            )
            ax.add_collection(pc)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_axis_off()

        t_txt = _fmt_dt(t_val)
        ax.set_title(f"{case_label} — h — {t_txt}", fontsize=16, fontweight="bold")

        cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("h", fontsize=12)

        tag = t_txt.replace(":", "-")
        out_png = os.path.join(out_dir, f"{case_label}_h_{init_tag}_{tag}_zoom.png".replace("__", "_"))
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ saved {out_png}")

    ds.close()
