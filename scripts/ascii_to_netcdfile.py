#!/usr/bin/env python3
import os
import sys
import argparse
import xarray as xr  # only used for optional post-write check/transpose

# Make src importable
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

# Import your function from your module
from flow_depth_plotting import ascii_single_member_to_netcdf  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=(
            "Convert LISFLOOD ASCII grids to NetCDF. "
            "Optionally compute derived velocity/discharge fields."
        )
    )

    # Base+range
    p.add_argument("--dir", required=True, help="Directory containing the ASCII files")
    p.add_argument("--base", required=True, help="Base name (prefix before the dash), e.g. 'Zell_2m'")
    p.add_argument("--start", type=int, required=True, help="Start index, e.g. 0 for 0000")
    p.add_argument("--end", type=int, required=True, help="End index (inclusive), e.g. 12 for 0012")
    p.add_argument("--width", type=int, default=4, help="Zero-pad width for indices (default: 4)")

    # Output & I/O
    p.add_argument("--out", required=True, help="Output NetCDF path")
    p.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    p.add_argument("--chunks", default=None, help="Dask chunks as 'X,Y' -> {'x':X,'y':Y} (e.g. 2000,2000)")

    # Data options
    p.add_argument("--crs", default="EPSG:2056", help="CRS to write (default: EPSG:2056)")
    p.add_argument("--var", default="water_depth", help="Variable name for wd (default: water_depth)")
    p.add_argument("--nodata", type=float, default=None, help="Override nodata; default uses file header")
    p.add_argument("--dtype", default="float32", help="Output dtype (default: float32)")
    p.add_argument("--complevel", type=int, default=4, help="zlib compression level 0–9 (default: 4)")
    p.add_argument(
        "--regex",
        default=r"-(\d{4})\.(?:\w+)$",
        help=r"Regex to extract index (default captures 4-digit index like '-0012.*')",
    )

    # Time dimension + timestamps (NEW)
    p.add_argument(
        "--dim-name",
        default="REFERENCE_TS",
        help="Name of stacked time dimension in NetCDF (default: REFERENCE_TS)",
    )
    p.add_argument(
        "--reference-start",
        required=True,
        help="Timestamp for the FIRST READ FILE (index=start). Example: 2022-05-05T12:00:00.000000000",
    )
    p.add_argument(
        "--dt-minutes",
        type=int,
        default=60,
        help="Minutes between timesteps (default: 60 for hourly)",
    )

    # Derived fields
    p.add_argument(
        "--compute-derived",
        action="store_true",
        help="Also read Vx/Vy/Qx/Qy, center them to cell grid, and compute derived fields.",
    )
    p.add_argument(
        "--wet-threshold",
        type=float,
        default=0.0,
        help="Mask derived fields where wd < threshold (default: 0.0)",
    )
    p.add_argument(
        "--realization",
        type=int,
        default=None,
        help="If set, include a 'realization' dim with this integer value.",
    )

    # Extensions (change if your filenames differ)
    p.add_argument("--ext-wd", default="wd", help="Extension for water depth files (default: wd)")
    p.add_argument("--ext-vx", default="Vx", help="Extension for velocity x-face files (default: Vx)")
    p.add_argument("--ext-vy", default="Vy", help="Extension for velocity y-face files (default: Vy)")
    p.add_argument("--ext-qx", default="Qx", help="Extension for discharge x-face files (default: Qx)")
    p.add_argument("--ext-qy", default="Qy", help="Extension for discharge y-face files (default: Qy)")

    # Optional: store arrays as (DIM,x,y)/(x,y) instead of (DIM,y,x)/(y,x)
    p.add_argument("--order-xy", action="store_true",
                   help="Force variable dims to (DIM,x,y) (or (x,y) if single slice)")

    args = p.parse_args()

    # Prepare var_map: always include wd; add others only if --compute-derived
    var_map = {"water_depth": args.ext_wd}
    if args.compute_derived:
        var_map.update({
            "vel_x": args.ext_vx,
            "vel_y": args.ext_vy,
            "flux_x": args.ext_qx,
            "flux_y": args.ext_qy,
        })

    # Chunks
    chunks = None
    if args.chunks:
        try:
            x_s, y_s = args.chunks.split(",")
            chunks = {"x": int(x_s), "y": int(y_s)}
        except Exception:
            p.error("Invalid --chunks. Use 'X,Y' like 2000,2000")

    # Overwrite if requested
    if args.force and os.path.exists(args.out):
        try:
            os.remove(args.out)
        except Exception as e:
            p.error(f"Could not remove existing output '{args.out}': {e}")

    # Call the writer
    out = ascii_single_member_to_netcdf(
        out_nc=args.out,
        folder=args.dir,
        base=args.base,
        start=args.start,
        end=args.end,
        width=args.width,
        var_map=var_map,
        crs=args.crs,
        nodata=args.nodata,
        dtype=args.dtype,
        complevel=args.complevel,
        chunks=chunks,
        step_regex=args.regex,
        strict_align=True,
        use_nan_fill=False,
        wet_threshold=args.wet_threshold,
        skip_missing=False,
        dim_name=args.dim_name,
        reference_start=args.reference_start,
        dt_minutes=args.dt_minutes,
        realization=args.realization,
    )
    print(f"Saved: {out}")

    # Optional reorder
    if args.order_xy:
        ds = xr.open_dataset(args.out)
        var = args.var if args.var in ds else "water_depth"
        if var not in ds:
            ds.close()
            raise KeyError(f"Variable '{var}' not found in {args.out}")

        dim = args.dim_name
        needs_write = False

        if dim in ds[var].dims and ds[var].ndim == 3:
            if tuple(ds[var].dims) != (dim, "x", "y"):
                ds[var] = ds[var].transpose(dim, "x", "y")
                needs_write = True
        else:
            if tuple(ds[var].dims) != ("x", "y"):
                ds[var] = ds[var].transpose("x", "y")
                needs_write = True

        if needs_write:
            tmp_path = args.out + ".tmp"
            ds.to_netcdf(tmp_path, mode="w")
            ds.close()
            os.replace(tmp_path, args.out)
            print(f"Rewritten with (x,y) as last dims: {args.out}")
        else:
            ds.close()
            print("No reorder needed; dims already in desired order.")


if __name__ == "__main__":
    main()


    
    
    