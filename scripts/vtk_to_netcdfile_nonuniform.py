#!/usr/bin/env python3
import os, re, sys, glob, shutil, subprocess
import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import xarray as xr

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)

# =============================================================================
# ARGUMENTS
# =============================================================================
if len(sys.argv) < 4:
    print("Usage: python vtk_to_ugrid_netcdf_vtu_pipeline.py <vtk_dir> <out.nc> <start_time> [freq] [epsg]")
    sys.exit(1)

INPUT_DIR  = sys.argv[1]
OUT_NC     = sys.argv[2]
START_TIME = sys.argv[3]
FREQ       = sys.argv[4] if len(sys.argv) >= 5 else "1h"
EPSG       = int(sys.argv[5]) if len(sys.argv) >= 6 else 2056

VTK_PATTERN = "Zell_50cm-*.vtk"

H_NAME  = "h"
Z0_NAME = "z0"

# IMPORTANT: DO NOT set NaN->0; keep missingness.
NODATA = -9999.0


# =============================================================================
# HELPERS
# =============================================================================
def timestep_from_name(path):
    m = re.search(r"-(\d+)\.vtk$", os.path.basename(path))
    if not m:
        m = re.search(r"-(\d+)_san\.vtk$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse timestep from filename: {path}")
    return int(m.group(1))


def sanitize_vtk_nan_to_nodata(vtk_path, nodata=NODATA):
    """
    Replace literal 'nan' tokens with a NODATA flag in ASCII legacy VTK.
    """
    base, ext = os.path.splitext(vtk_path)
    out = base + "_san" + ext

    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(vtk_path):
        return out

    nan_re = re.compile(r"\bnan\b", flags=re.IGNORECASE)
    with open(vtk_path, "r", errors="ignore") as f, open(out, "w") as g:
        for line in f:
            g.write(nan_re.sub(str(nodata), line))
    return out


def ensure_vtu(vtk_path):
    """
    Convert legacy VTK -> VTU using pvpython (ParaView), cached.
    Writes <samebase>.vtu next to input.
    """
    pvpython = shutil.which("pvpython")
    if pvpython is None:
        raise RuntimeError("pvpython not found. Load ParaView module or ensure pvpython is on PATH.")

    base, _ = os.path.splitext(vtk_path)
    vtu = base + ".vtu"

    if os.path.exists(vtu) and os.path.getmtime(vtu) >= os.path.getmtime(vtk_path):
        return vtu

    code = f"""
from paraview.simple import *
s = OpenDataFile(r"{vtk_path}")
SaveData(r"{vtu}", proxy=s)
print("WROTE_OK")
"""
    proc = subprocess.run([pvpython, "-c", code],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or "WROTE_OK" not in proc.stdout:
        raise RuntimeError(
            f"pvpython conversion failed.\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    return vtu


def read_vtu(path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    ug = r.GetOutput()
    if ug is None or ug.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Empty/invalid VTU: {path}")
    return ug


def nodata_to_nan(a, nodata=NODATA):
    a = a.astype(np.float64, copy=False)
    return np.where(a <= (nodata + 1.0), np.nan, a)


def extract_points_xy(ug):
    pts = ug.GetPoints()
    arr = vtk_to_numpy(pts.GetData())  # (npoints, 3)
    return arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)


def extract_face_node_connectivity(ug, fill_value=-1):
    """
    Fixed-width connectivity (nFaces, max_nodes_per_face), padded with -1.
    Works for mixed triangles/quads/polygons.
    """
    nfaces = ug.GetNumberOfCells()
    max_n = 0
    for ci in range(nfaces):
        max_n = max(max_n, ug.GetCell(ci).GetNumberOfPoints())

    conn = np.full((nfaces, max_n), fill_value, dtype=np.int64)
    for ci in range(nfaces):
        ids = ug.GetCell(ci).GetPointIds()
        nn = ids.GetNumberOfIds()
        for j in range(nn):
            conn[ci, j] = ids.GetId(j)

    return conn, max_n


def compute_face_centroids(ug):
    nfaces = ug.GetNumberOfCells()
    cx = np.empty(nfaces, dtype=np.float64)
    cy = np.empty(nfaces, dtype=np.float64)
    pts = ug.GetPoints()

    for ci in range(nfaces):
        ids = ug.GetCell(ci).GetPointIds()
        xs = []
        ys = []
        for j in range(ids.GetNumberOfIds()):
            pid = ids.GetId(j)
            px, py, _ = pts.GetPoint(pid)
            xs.append(px)
            ys.append(py)
        cx[ci] = np.mean(xs) if xs else np.nan
        cy[ci] = np.mean(ys) if ys else np.nan

    return cx, cy


def get_array_from_cell_or_point(ug, name):
    """
    Returns ('cell'|'node', numpy_array) or (None, None).
    """
    a = ug.GetCellData().GetArray(name)
    if a is not None:
        return "cell", vtk_to_numpy(a)

    a = ug.GetPointData().GetArray(name)
    if a is not None:
        return "node", vtk_to_numpy(a)

    return None, None


# =============================================================================
# MAIN
# =============================================================================
files = sorted(
    [f for f in glob.glob(os.path.join(INPUT_DIR, VTK_PATTERN)) if "_san" not in f],
    key=timestep_from_name
)
if not files:
    raise RuntimeError(f"No VTK files found in {INPUT_DIR} matching {VTK_PATTERN}")

time = pd.date_range(START_TIME, periods=len(files), freq=FREQ).to_numpy(dtype="datetime64[ns]")

# --- define mesh from first timestep
f0_san = sanitize_vtk_nan_to_nodata(files[0], NODATA)
f0_vtu = ensure_vtu(f0_san)
ug0 = read_vtu(f0_vtu)

x_node, y_node = extract_points_xy(ug0)
face_nodes, max_nodes = extract_face_node_connectivity(ug0, fill_value=-1)
x_face, y_face = compute_face_centroids(ug0)

n_node = x_node.size
n_face = face_nodes.shape[0]

# --- decide where h lives (cell vs node)
h_loc0, _ = get_array_from_cell_or_point(ug0, H_NAME)
if h_loc0 is None:
    raise RuntimeError(f"Array '{H_NAME}' not found in cell or point data in {files[0]}")

# optional z0
z0_loc0, z00 = get_array_from_cell_or_point(ug0, Z0_NAME)

# allocate time varying h
if h_loc0 == "cell":
    H = np.full((len(files), n_face), np.nan, dtype=np.float32)
else:
    H = np.full((len(files), n_node), np.nan, dtype=np.float32)

# store z0 once if present
z0 = None
z0_loc = None
if z0_loc0 is not None:
    z0_loc = z0_loc0
    z0 = nodata_to_nan(z00, NODATA).astype(np.float32)
    if z0_loc == "cell" and z0.size != n_face:
        raise RuntimeError(f"z0 size mismatch: got {z0.size}, expected {n_face} (cell)")
    if z0_loc == "node" and z0.size != n_node:
        raise RuntimeError(f"z0 size mismatch: got {z0.size}, expected {n_node} (node)")

# --- loop timesteps
for i, f in enumerate(files):
    print(f"[{i+1}/{len(files)}] {os.path.basename(f)}")

    san = sanitize_vtk_nan_to_nodata(f, NODATA)
    vtu = ensure_vtu(san)
    ug = read_vtu(vtu)

    # topology consistency check
    if ug.GetNumberOfPoints() != n_node or ug.GetNumberOfCells() != n_face:
        raise RuntimeError(
            f"Mesh changed at {f}: points {ug.GetNumberOfPoints()} vs {n_node}, "
            f"cells {ug.GetNumberOfCells()} vs {n_face}."
        )

    h_loc, h_arr = get_array_from_cell_or_point(ug, H_NAME)
    if h_loc != h_loc0:
        raise RuntimeError(f"'{H_NAME}' location changed (was {h_loc0}, now {h_loc}) at {f}")

    h_arr = nodata_to_nan(h_arr, NODATA).astype(np.float32)

    if h_loc0 == "cell":
        if h_arr.size != n_face:
            raise RuntimeError(f"h size mismatch at {f}: got {h_arr.size}, expected {n_face}")
        H[i, :] = h_arr
    else:
        if h_arr.size != n_node:
            raise RuntimeError(f"h size mismatch at {f}: got {h_arr.size}, expected {n_node}")
        H[i, :] = h_arr

    del ug

# =============================================================================
# WRITE NETCDF (UGRID-style)
# =============================================================================
mesh_name = "mesh2d"

coords = {
    "time": ("time", time),
    "mesh2d_node": ("mesh2d_node", np.arange(n_node, dtype=np.int64)),
    "mesh2d_face": ("mesh2d_face", np.arange(n_face, dtype=np.int64)),
    "max_n_face_nodes": ("max_n_face_nodes", np.arange(max_nodes, dtype=np.int64)),
}

data_vars = {
    mesh_name: xr.DataArray(
        np.int32(1),
        attrs={
            "cf_role": "mesh_topology",
            "topology_dimension": 2,
            "node_coordinates": "mesh2d_node_x mesh2d_node_y",
            "face_node_connectivity": "mesh2d_face_nodes",
        },
    ),
    "mesh2d_node_x": (("mesh2d_node",), x_node.astype(np.float64)),
    "mesh2d_node_y": (("mesh2d_node",), y_node.astype(np.float64)),
    "mesh2d_face_nodes": (("mesh2d_face", "max_n_face_nodes"), face_nodes),
    "mesh2d_face_x": (("mesh2d_face",), x_face.astype(np.float64)),
    "mesh2d_face_y": (("mesh2d_face",), y_face.astype(np.float64)),
}

# add h
if h_loc0 == "cell":
    data_vars["h"] = (("time", "mesh2d_face"), H)
else:
    data_vars["h"] = (("time", "mesh2d_node"), H)

# add z0 if present
if z0 is not None:
    if z0_loc == "cell":
        data_vars["z0"] = (("mesh2d_face",), z0)
    else:
        data_vars["z0"] = (("mesh2d_node",), z0)

ds = xr.Dataset(
    data_vars=data_vars,
    coords=coords,
    attrs={
        "crs_epsg": EPSG,
        "source_vtk_pattern": VTK_PATTERN,
        "pipeline": "legacy VTK -> sanitize NaN token -> VTU (pvpython) -> NetCDF (UGRID)",
        "nodata_flag_used_in_sanitization": NODATA,
        "note": "Unstructured mesh preserved exactly; no interpolation/resampling performed.",
    }
)

# ---- FIX: set attrs/encoding AFTER dataset creation
ds["mesh2d_face_nodes"].attrs["start_index"] = 0
ds["mesh2d_face_nodes"].encoding["_FillValue"] = np.int64(-1)

# helpful UGRID metadata for variables
ds["h"].attrs["mesh"] = mesh_name
ds["h"].attrs["location"] = "face" if h_loc0 == "cell" else "node"
if "z0" in ds:
    ds["z0"].attrs["mesh"] = mesh_name
    ds["z0"].attrs["location"] = "face" if z0_loc == "cell" else "node"

encoding = {
    "h": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "mesh2d_face_nodes": {"zlib": True, "complevel": 4, "dtype": "int64"},
    "mesh2d_node_x": {"zlib": True, "complevel": 4, "dtype": "float64"},
    "mesh2d_node_y": {"zlib": True, "complevel": 4, "dtype": "float64"},
    "mesh2d_face_x": {"zlib": True, "complevel": 4, "dtype": "float64"},
    "mesh2d_face_y": {"zlib": True, "complevel": 4, "dtype": "float64"},
}
if "z0" in ds:
    encoding["z0"] = {"zlib": True, "complevel": 4, "dtype": "float32"}

os.makedirs(os.path.dirname(OUT_NC) or ".", exist_ok=True)
tmp_nc = OUT_NC + ".tmp"
ds.to_netcdf(tmp_nc, encoding=encoding)
os.replace(tmp_nc, OUT_NC)

print("WROTE:", OUT_NC)
print(f"Mesh: nodes={n_node}, faces={n_face}, max_nodes_per_face={max_nodes}")
print(f"h stored on: {h_loc0}")

# =============================================================================
# CLEANUP TEMP FILES (_san.vtk and _san.vtu)
# =============================================================================
patterns_to_remove = [
    os.path.join(INPUT_DIR, "*_san.vtk"),
    os.path.join(INPUT_DIR, "*_san.vtu"),
]
removed = 0
for pattern in patterns_to_remove:
    for fp in glob.glob(pattern):
        try:
            os.remove(fp)
            removed += 1
        except Exception as e:
            print(f"[WARN] Could not delete {fp}: {e}")

print(f"Cleanup done: removed {removed} temporary *_san.vtk / *_san.vtu files.")

