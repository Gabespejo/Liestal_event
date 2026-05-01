import os
import subprocess
from osgeo import gdal, ogr

def rasterize_buildings(buildings_vector, dtm_tif, out_buildings_mask_tif,
                        burn_value=1, background_value=0):

    os.makedirs(os.path.dirname(out_buildings_mask_tif), exist_ok=True)

    # remove existing output
    if os.path.exists(out_buildings_mask_tif):
        os.remove(out_buildings_mask_tif)

    # open dtm to get grid
    dtm = gdal.Open(dtm_tif)
    if dtm is None:
        raise FileNotFoundError(f"Cannot open DTM: {dtm_tif}")

    gt = dtm.GetGeoTransform()
    xsize, ysize = dtm.RasterXSize, dtm.RasterYSize

    xmin = gt[0]
    ymax = gt[3]
    xres = gt[1]
    yres = abs(gt[5])
    xmax = xmin + xsize * xres
    ymin = ymax - ysize * yres

    # get layer name (works for GeoJSON/SHP/GPKG)
    vds = ogr.Open(buildings_vector)
    if vds is None or vds.GetLayerCount() < 1:
        raise RuntimeError(f"Cannot open buildings vector or no layers: {buildings_vector}")
    layer_name = vds.GetLayer(0).GetName()
    vds = None

    # build CLI command
    cmd = [
        "gdal_rasterize",
        "-burn", str(burn_value),
        "-a_nodata", str(background_value),
        "-l", layer_name,
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-tr", str(xres), str(yres),
        "-tap",
        "-ot", "Byte",
        "-co", "COMPRESS=LZW",
        buildings_vector,
        out_buildings_mask_tif
    ]

    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        raise RuntimeError(
            "gdal_rasterize failed.\n"
            f"STDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr}\n"
        )

    print(f"✅ Buildings mask saved: {out_buildings_mask_tif}")



#############################################################################################
########COMBINE DTM AND DSM FILES ###########################################################

import numpy as np
from osgeo import gdal
import os

def DTM_DSM_both(
    dtm_tif,
    dsm_tif,
    buildings_mask_tif,
    out_dem_mix_tif,
    resampleAlg="max",
    nodata=-9999
):
    """
    Create DEM_mix:
      DEM_mix = DSM where buildings=1, else DTM
    Steps:
      1) Warp DSM to match DTM grid (extent/resolution/alignment)
      2) Apply the equation using buildings mask
    """
    os.makedirs(os.path.dirname(out_dem_mix_tif), exist_ok=True)

    # --- Open DTM to get reference grid ---
    dtm_ds = gdal.Open(dtm_tif)
    if dtm_ds is None:
        raise FileNotFoundError(f"Cannot open DTM: {dtm_tif}")

    gt = dtm_ds.GetGeoTransform()
    proj = dtm_ds.GetProjection()
    xsize, ysize = dtm_ds.RasterXSize, dtm_ds.RasterYSize

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * xsize
    ymin = ymax + gt[5] * ysize  # gt[5] negative

    xRes = abs(gt[1])
    yRes = abs(gt[5])

    # --- Step 1: Warp DSM to match DTM grid exactly ---
    dsm_warped_path = out_dem_mix_tif.replace(".tif", "_DSM_warped_to_DTM.tif")

    gdal.Warp(
        dsm_warped_path,
        dsm_tif,
        format="GTiff",
        dstSRS=proj,
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=xRes,
        yRes=yRes,
        targetAlignedPixels=True,
        resampleAlg=resampleAlg,
        dstNodata=nodata,
        multithread=True
    )

    # --- Read arrays ---
    dtm = dtm_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    bld_ds = gdal.Open(buildings_mask_tif)
    if bld_ds is None:
        raise FileNotFoundError(f"Cannot open buildings mask: {buildings_mask_tif}")
    bld = bld_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

    dsm_ds = gdal.Open(dsm_warped_path)
    if dsm_ds is None:
        raise FileNotFoundError(f"Cannot open warped DSM: {dsm_warped_path}")
    dsm = dsm_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # --- Step 2: Apply thesis equation (clear form) ---
    # DEM_mix = DSM*buildings + DTM*(1-buildings)
    out = (dsm * bld) + (dtm * (1 - bld))

    # --- Write output ---
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(out_dem_mix_tif, xsize, ysize, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(out)
    out_band.SetNoDataValue(nodata)
    out_band.FlushCache()

    # Close datasets
    out_ds = None
    dtm_ds = None
    bld_ds = None
    dsm_ds = None

    print(f"✅ DEM_mix saved: {out_dem_mix_tif}")
    print(f"(Intermediate DSM warped to DTM grid: {dsm_warped_path})")
