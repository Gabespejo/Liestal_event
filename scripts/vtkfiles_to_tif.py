#!/usr/bin/env -S mamba run -n env_py311 python
import os, sys, subprocess, shutil
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import rasterio
from rasterio.transform import from_origin

def usage():
    print("Usage:")
    print("  python vtk_to_tif_lv95_force_pvpython.py <input.vtk> <variable> <output.tif> [dx] [epsg]")
    sys.exit(1)

if len(sys.argv) < 4:
    usage()

vtk_path = sys.argv[1]
varname  = sys.argv[2]
tif_path = sys.argv[3]
dx = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.5
epsg = int(sys.argv[5]) if len(sys.argv) >= 6 else 2056

os.makedirs(os.path.dirname(tif_path) or ".", exist_ok=True)

# Put VTU next to the VTK (same name)
vtu_path = os.path.splitext(vtk_path)[0] + ".vtu"

# ---- 1) Convert VTK -> VTU using pvpython (ParaView reader) ----
pvpython = shutil.which("pvpython")
if pvpython is None:
    raise RuntimeError(
        "pvpython not found. Run:\n"
        "  module load ParaView/5.11.2-foss-2023a\n"
        "and then run this script in the same shell (NOT via conda run)."
    )

pv_code = f"""
from paraview.simple import *
src = OpenDataFile(r"{vtk_path}")
SaveData(r"{vtu_path}", proxy=src)
Delete(src)
print("WROTE_OK")
"""

print("[1/3] Converting VTK -> VTU using pvpython")
proc = subprocess.run([pvpython, "-c", pv_code], text=True,
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if proc.returncode != 0 or "WROTE_OK" not in proc.stdout:
    raise RuntimeError("pvpython conversion failed.\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr)

print("  VTU:", vtu_path)

# ---- 2) Read VTU with VTK XML reader (stable) ----
print("[2/3] Reading VTU + resampling to uniform grid")
r = vtk.vtkXMLUnstructuredGridReader()
r.SetFileName(vtu_path)
r.Update()
ug = r.GetOutput()
if ug is None or ug.GetNumberOfPoints() == 0:
    raise RuntimeError("VTU read produced empty dataset")

xmin, xmax, ymin, ymax, *_ = ug.GetBounds()
nx = int(np.floor((xmax - xmin) / dx)) + 1
ny = int(np.floor((ymax - ymin) / dx)) + 1

# IMPORTANT on your system: pipeline connection
producer = vtk.vtkTrivialProducer()
producer.SetOutput(ug)

res = vtk.vtkResampleToImage()
res.SetInputConnection(producer.GetOutputPort())
res.SetSamplingBounds(xmin, xmax, ymin, ymax, 0.0, 0.0)
res.SetSamplingDimensions(nx, ny, 1)
res.Update()

img = res.GetOutput()
if img is None:
    raise RuntimeError("ResampleToImage produced no output")

# ---- 3) Extract variable + write GeoTIFF with EPSG:2056 ----
print("[3/3] Writing GeoTIFF with EPSG:%d" % epsg)

pd = img.GetPointData()
cd = img.GetCellData()

arr = pd.GetArray(varname) if pd and pd.GetArray(varname) else None
is_point = True
if arr is None and cd and cd.GetArray(varname):
    arr = cd.GetArray(varname)
    is_point = False

if arr is None:
    avail = []
    if pd: avail += [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    if cd: avail += [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]
    raise KeyError(f"Variable '{varname}' not found after pvpython conversion. Available: {avail}")

a = vtk_to_numpy(arr)

if is_point and a.size == nx * ny:
    data = a.reshape((ny, nx), order="C")
elif (not is_point) and a.size == (nx - 1) * (ny - 1):
    tmp = a.reshape((ny - 1, nx - 1), order="C")
    data = np.full((ny, nx), np.nan, dtype=tmp.dtype)
    data[:ny-1, :nx-1] = tmp
else:
    raise ValueError(f"Unexpected array size {a.size} for grid {nx}x{ny}")

# ✅ FIX: VTK image memory is bottom-up (ymin first), GeoTIFF expects top-down (ymax first)
data = np.flipud(data)

data = data.astype(np.float32)

# GeoTIFF north-up transform (top-left corner at xmin, ymax)
transform = from_origin(xmin, ymax, dx, dx)

with rasterio.open(
    tif_path, "w",
    driver="GTiff",
    height=ny, width=nx,
    count=1,
    dtype=data.dtype,
    crs=f"EPSG:{epsg}",
    transform=transform,
    compress="deflate",
    tiled=True,
) as dst:
    dst.write(data, 1)

print(" DONE")
print("GeoTIFF:", tif_path)
print("VTU    :", vtu_path)
print(f"Grid   : nx={nx}, ny={ny}, dx={dx}")
print(f"Bounds : xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")




