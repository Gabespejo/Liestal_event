#!/usr/bin/env -S mamba run -n env_py311 python
# Run from terminal; imports function from src/flow_depth_plotting.py

import sys, os, argparse

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

from flow_depth_plotting import ascii_stack_to_netcdf  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description="Convert ESRI ASCII grids (*.wd/*.asc) to NetCDF via flow_depth_plotting.ascii_stack_to_netcdf"
    )

    # EITHER: pattern mode
    p.add_argument("--pattern", help="Glob or file, e.g. 'Zell_2m-*.wd' or 'Zell_2m-0004.wd'")

    # OR: base+range mode (saves in the same directory by default)
    p.add_argument("--dir", help="Directory containing the .wd files")
    p.add_argument("--base", help="Base name (prefix before the dash), e.g. 'Zell_2m_bach'")
    p.add_argument("--start", type=int, help="Start index, e.g. 0 for 0000")
    p.add_argument("--end", type=int, help="End index (inclusive), e.g. 12 for 0012")
    p.add_argument("--width", type=int, default=4, help="Zero-pad width for indices (default: 4)")

    # Common options
    p.add_argument("--out", default=None,
                   help="Output NetCDF path. If omitted in base+range mode, defaults to '<dir>/<base>_wd_<start..end>.nc'")
    p.add_argument("--crs", default="EPSG:2056", help="CRS to write (default: EPSG:2056)")
    p.add_argument("--var", default="water_depth", help="Variable name (default: water_depth)")
    p.add_argument("--nodata", type=float, default=None, help="Override nodata; default uses file header")
    p.add_argument("--dtype", default="float32", help="Output dtype (default: float32)")
    p.add_argument("--complevel", type=int, default=4, help="zlib compression level 0–9 (default: 4)")
    p.add_argument("--regex", default=r"-(\d{4})\.(?:wd|asc)$",
                   help=r"Regex to extract step index (default captures 4-digit index like '-0012.wd')")
    p.add_argument("--chunks", default=None,
                   help="Dask chunks as 'X,Y' -> {'x':X,'y':Y}. Example: 2000,2000")
    p.add_argument("--nan-fill", action="store_true", help="Store NaNs instead of a fixed _FillValue")
    p.add_argument("--no-strict-align", action="store_true", help="Disable strict geotransform/shape checks")
    p.add_argument("--skip-missing", action="store_true",
                   help="Skip missing files in base+range mode (default: error if any missing)")

    args = p.parse_args()

    # Build inputs
    if args.pattern:
        inputs = args.pattern
        if args.out is None:
            p.error("--out is required when using --pattern")
        out_path = args.out
    else:
        # Base+range mode
        if not (args.dir and args.base and args.start is not None and args.end is not None):
            p.error("Provide either --pattern OR (--dir --base --start --end)")
        if args.start > args.end:
            p.error("--start must be <= --end")

        # Build explicit list of files with zero-padding
        inputs = []
        missing = []
        for i in range(args.start, args.end + 1):
            fname = f"{args.base}-{i:0{args.width}d}.wd"
            fpath = os.path.join(args.dir, fname)
            if os.path.exists(fpath):
                inputs.append(fpath)
            else:
                missing.append(fpath)

        if missing and not args.skip_missing:
            p.error("Missing files:\n  " + "\n  ".join(missing))
        if not inputs:
            p.error("No existing files found for the requested range.")

        # Default output path in the same directory if not provided
        out_path = args.out or os.path.join(
            args.dir,
            f"{args.base}_wd_{args.start:0{args.width}d}-{args.end:0{args.width}d}.nc"
        )

    # Chunks
    chunks = None
    if args.chunks:
        try:
            x_s, y_s = args.chunks.split(",")
            chunks = {"x": int(x_s), "y": int(y_s)}
        except Exception:
            p.error("Invalid --chunks. Use 'X,Y' like 2000,2000")

    out = ascii_stack_to_netcdf(
        inputs=inputs,
        out_nc=out_path,
        crs=args.crs,
        var_name=args.var,
        nodata=args.nodata,
        dtype=args.dtype,
        complevel=args.complevel,
        step_regex=args.regex,
        chunks=chunks,
        use_nan_fill=args.nan_fill,
        strict_align=not args.no_strict_align,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()