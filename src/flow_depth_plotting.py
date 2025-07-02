

########## PLOTTING ############################################ 

import os
import numpy as np
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