#!/usr/bin/env python3
import os, re, sys, glob, shutil, subprocess
import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import xarray as xr


def usage():
    print("Usage:")
    print("  python vtk_series_to_netcdf.py <input_vtk_dir> <output_nc_file> <dem_path|NONE> <start_time> [freq] [dx] [epsg]")
    print("")
    print("Examples:")
    print("  # Use DEM bounds")
    print("  python vtk_series_to_netcdf.py /path/vtks out.nc /path/domain.dem 2022-05-05T12:00:00 1h 0.5 2056")
    print("")
    print("  # Use full VTK extent (no DEM cropping)")
    print("  python vtk_series_to_netcdf.py /path/vtks out.nc NONE 2022-05-05T12:00:00 1h 0.5 2056")
    sys.exit(1)


if len(sys.argv) < 5:
    usage()

INPUT_DIR   = sys.argv[1]
OUT_NC      = sys.argv[2]
DEM_PATH_IN = sys.argv[3]
START_TIME  = sys.argv[4]
FREQ        = sys.argv[5] if len(sys.argv) >= 6 else "1h"
DX          = float(sys.argv[6]) if len(sys.argv) >= 7 else 0.5
EPSG        = int(sys.argv[7]) if len(sys.argv) >= 8 else 2056

DEM_PATH = None
if DEM_PATH_IN and DEM_PATH_IN.strip().upper() != "NONE":
    DEM_PATH = DEM_PATH_IN

VTK_PATTERN = "Zell_50cm-*.vtk"
H_NAME = "h"
Z0_NAME = "z0"

# ✅ Use nodata instead of turning nan into 0 (prevents "false dry")
NODATA_VALUE = -9999.0


def timestep_from_name(path):
    base = os.path.basename(path)
    m = re.search(r"-(\d+)\.vtk$", base)
    if m:
        return int(m.group(1))
    m = re.search(r"-(\d+)_san\.vtk$", base)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse timestep from filename: {base}")


def read_dem_bounds(dem_path: str):
    """Read ESRI ASCII grid header (*.dem) and return (xmin, xmax, ymin, ymax)."""
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    header = {}
    with open(dem_path, "r") as f:
        for _ in range(6):
            line = f.readline()
            if not line:
                raise ValueError(f"DEM header too short: {dem_path}")
            parts = line.strip().split()
            if len(parts) < 2:
                raise ValueError(f"Bad DEM header line: {line!r}")
            header[parts[0].lower()] = float(parts[1])

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cell  = header["cellsize"]

    xmin = header["xllcorner"]
    ymin = header["yllcorner"]
    xmax = xmin + ncols * cell
    ymax = ymin + nrows * cell

    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Invalid DEM-derived bounds from {dem_path}: {(xmin, xmax, ymin, ymax)}")

    return (xmin, xmax, ymin, ymax)


def sanitize_vtk_nan_to_nodata(vtk_path: str, nodata: float = NODATA_VALUE) -> str:
    """Replace tokens 'nan' with a nodata value in ASCII legacy VTK. Output cached as *_san.vtk."""
    base, ext = os.path.splitext(vtk_path)
    out_path = base + "_san" + ext

    if os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(vtk_path):
        return out_path

    nan_re = re.compile(r"\bnan\b", flags=re.IGNORECASE)
    with open(vtk_path, "r", errors="ignore") as fin, open(out_path, "w") as fout:
        for line in fin:
            fout.write(nan_re.sub(str(nodata), line))

    return out_path


def ensure_vtu_with_pvpython(vtk_path: str) -> str:
    """Convert legacy VTK -> VTU using pvpython (ParaView), cached."""
    pvpython = shutil.which("pvpython")
    if pvpython is None:
        raise RuntimeError(
            "pvpython not found. Load ParaView, e.g.\n"
            "  module load ParaView/5.11.2-foss-2023a\n"
        )

    base, _ = os.path.splitext(vtk_path)
    vtu_path = base + ".vtu"

    if os.path.exists(vtu_path) and os.path.getmtime(vtu_path) >= os.path.getmtime(vtk_path):
        return vtu_path

    pv_code = f"""
from paraview.simple import *
src = OpenDataFile(r"{vtk_path}")
SaveData(r"{vtu_path}", proxy=src)
Delete(src)
print("WROTE_OK")
"""
    proc = subprocess.run([pvpython, "-c", pv_code], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0 or "WROTE_OK" not in proc.stdout:
        raise RuntimeError("pvpython conversion failed.\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr)

    return vtu_path


def read_vtu(vtu_path: str):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(vtu_path)
    r.Update()
    ug = r.GetOutput()
    if ug is None or ug.GetNumberOfPoints() == 0:
        raise RuntimeError(f"VTU read produced empty dataset: {vtu_path}")
    return r, ug


def bounds_from_ug_xy(ug):
    """Return (xmin, xmax, ymin, ymax) from an UnstructuredGrid bounds."""
    xmin, xmax, ymin, ymax, zmin, zmax = ug.GetBounds()
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Invalid VTK bounds: {(xmin, xmax, ymin, ymax)}")
    return (float(xmin), float(xmax), float(ymin), float(ymax))


def resample_to_image(reader_output_port, bounds_xy, dx):
    xmin, xmax, ymin, ymax = bounds_xy
    nx = int(np.floor((xmax - xmin) / dx)) + 1
    ny = int(np.floor((ymax - ymin) / dx)) + 1

    res = vtk.vtkResampleToImage()
    res.SetInputConnection(reader_output_port)
    res.SetSamplingBounds(xmin, xmax, ymin, ymax, 0.0, 0.0)
    res.SetSamplingDimensions(nx, ny, 1)
    res.Update()

    img = res.GetOutput()
    if img is None:
        raise RuntimeError("ResampleToImage produced no output.")
    return img, nx, ny


def extract_array_2d(img, name, nx, ny):
    """Try PointData first, then CellData."""
    pd_ = img.GetPointData()
    cd_ = img.GetCellData()

    arr = pd_.GetArray(name) if pd_ and pd_.GetArray(name) else None
    is_point = True
    if arr is None and cd_ and cd_.GetArray(name):
        arr = cd_.GetArray(name)
        is_point = False

    if arr is None:
        avail = []
        if pd_:
            avail += [pd_.GetArrayName(i) for i in range(pd_.GetNumberOfArrays())]
        if cd_:
            avail += [cd_.GetArrayName(i) for i in range(cd_.GetNumberOfArrays())]
        raise KeyError(f"Array '{name}' not found. Available: {avail}")

    a = vtk_to_numpy(arr)

    if is_point:
        if a.size != nx * ny:
            raise ValueError(f"Point array '{name}' size {a.size} != nx*ny {nx*ny}")
        data = a.reshape((ny, nx), order="C")
    else:
        if a.size != (nx - 1) * (ny - 1):
            raise ValueError(f"Cell array '{name}' size {a.size} != (nx-1)*(ny-1)")
        tmp = a.reshape((ny - 1, nx - 1), order="C")
        data = np.full((ny, nx), np.nan, dtype=tmp.dtype)
        data[:ny-1, :nx-1] = tmp

    return data


# -------------------------
# Main
# -------------------------
all_files = glob.glob(os.path.join(INPUT_DIR, VTK_PATTERN))
files = [f for f in all_files if not f.endswith("_san.vtk") and not f.endswith(".san.vtk")]
files = sorted(files, key=timestep_from_name)

if not files:
    raise RuntimeError(f"No original VTK files found in {INPUT_DIR} matching {VTK_PATTERN}")

# --- Determine bounds (DEM if provided, else VTK bounds)
bounds_xy = None
if DEM_PATH is not None:
    bounds_xy = read_dem_bounds(DEM_PATH)
    print("Using DEM bounds:", DEM_PATH)
    print("DEM-derived bounds_xy:", bounds_xy)
else:
    # Use first file to determine domain bounds
    f0_san = sanitize_vtk_nan_to_nodata(files[0])
    vtu0 = ensure_vtu_with_pvpython(f0_san)
    r0, ug0 = read_vtu(vtu0)
    bounds_xy = bounds_from_ug_xy(ug0)
    print("No DEM provided -> using full VTK bounds from first timestep")
    print("VTK-derived bounds_xy:", bounds_xy)

# Time axis
time = pd.date_range(START_TIME, periods=len(files), freq=FREQ).to_numpy(dtype="datetime64[ns]")

# Grid from bounds + dx
xmin, xmax, ymin, ymax = bounds_xy
nx = int(np.floor((xmax - xmin) / DX)) + 1
ny = int(np.floor((ymax - ymin) / DX)) + 1

x = np.linspace(xmin, xmax, nx, dtype=np.float64)
y = np.linspace(ymax, ymin, ny, dtype=np.float64)

H = np.empty((len(files), ny, nx), dtype=np.float32)
z0 = None

for i, f in enumerate(files):
    print(f"[{i+1}/{len(files)}] {os.path.basename(f)}")

    try:
        f_san = sanitize_vtk_nan_to_nodata(f)
        vtu = ensure_vtu_with_pvpython(f_san)

        r, ug = read_vtu(vtu)
        img, nx_i, ny_i = resample_to_image(r.GetOutputPort(), bounds_xy, DX)

        if nx_i != nx or ny_i != ny:
            raise RuntimeError(f"Grid mismatch: got {nx_i}x{ny_i}, expected {nx}x{ny}")

        h_raw = extract_array_2d(img, H_NAME, nx, ny)

        # ✅ convert nodata flag back to NaN before saving
        h_raw = np.where(h_raw == NODATA_VALUE, np.nan, h_raw)

        H[i] = np.flipud(h_raw).astype(np.float32)

        if z0 is None:
            try:
                z0_raw = extract_array_2d(img, Z0_NAME, nx, ny)
                z0_raw = np.where(z0_raw == NODATA_VALUE, np.nan, z0_raw)
                z0 = np.flipud(z0_raw).astype(np.float32)
                print("  Stored static z0.")
            except KeyError:
                print("  z0 not found -> skipping z0 output.")

    except Exception as e:
        print(f"[WARN] Failed on {os.path.basename(f)} -> filling with NaNs. Reason: {e}")
        H[i, :, :] = np.nan

# Write NetCDF
ds_out = xr.Dataset(
    data_vars={
        "h": (("time", "y", "x"), H),
        **({"z0": (("y", "x"), z0)} if z0 is not None else {}),
    },
    coords={
        "time": ("time", time),
        "y": ("y", y),
        "x": ("x", x),
    },
    attrs={
        "crs_epsg": EPSG,
        "dx": DX,
        "dy": DX,
        "nodata": NODATA_VALUE,
        "dem_path": DEM_PATH if DEM_PATH is not None else "",
        "bounds_xy": str(bounds_xy),
        "start_time_for_step_0000": START_TIME,
        "freq": FREQ,
        "note": "Legacy VTK sanitized (nan->NODATA) -> pvpython VTU -> VTK ResampleToImage -> NetCDF",
    }
)

encoding = {
    "h": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": NODATA_VALUE,
        "chunksizes": (1, min(512, ny), min(512, nx))
    }
}
if z0 is not None:
    encoding["z0"] = {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": NODATA_VALUE,
        "chunksizes": (min(512, ny), min(512, nx))
    }

os.makedirs(os.path.dirname(OUT_NC) or ".", exist_ok=True)
ds_out.to_netcdf(OUT_NC, encoding=encoding)

print(f"\nWROTE NetCDF: {OUT_NC}")
print(f"Grid: nx={nx}, ny={ny}, dx={DX}")
print(f"Bounds: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
print(f"Time[0] (timestep 0000): {np.datetime_as_string(time[0])}")

# -------------------------
# Cleanup sanitized temporary files
# -------------------------
cleanup_dir = INPUT_DIR
patterns_to_remove = [
    os.path.join(cleanup_dir, "*_san.vtk"),
    os.path.join(cleanup_dir, "*_san.vtu"),
]

removed = 0
for pattern in patterns_to_remove:
    for f in glob.glob(pattern):
        try:
            os.remove(f)
            removed += 1
        except Exception as e:
            print(f"[WARN] Could not delete {f}: {e}")

print(f"Cleanup done: removed {removed} temporary *_san.vtk / *_san.vtu files.")
