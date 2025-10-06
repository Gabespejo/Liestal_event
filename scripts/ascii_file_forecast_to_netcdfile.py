#!/usr/bin/env -S mamba run -n env_py311 python
import os, sys, argparse, tempfile, shutil
import xarray as xr

# make sure src/ (where ascii_ensemble_to_netcdf lives) is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from flow_depth_plotting import ascii_ensemble_to_netcdf

def main():
    ap = argparse.ArgumentParser(
        description="Pack LISFLOOD ASCII outputs from ALL realizations into ONE NetCDF"
    )
    ap.add_argument("--parent-dir", required=True, help="Folder containing realization subfolders")
    ap.add_argument("--base",       required=True, help="Base file prefix (e.g. Zell_2m)")
    ap.add_argument("--r-start",    type=int, required=True, help="First realization index (e.g., 1)")
    ap.add_argument("--r-end",      type=int, required=True, help="Last realization index (e.g., 11)")
    ap.add_argument("--start",      type=int, required=True, help="First time-step index (e.g., 1)")
    ap.add_argument("--end",        type=int, required=True, help="Last time-step index (e.g., 10)")
    ap.add_argument("--width",      type=int, default=4, help="Zero-pad width in filenames (default 4 for 0001)")
    ap.add_argument("--ens-suffix", default="fv1-gpu",
                    help="Realization folder suffix; expects folders like <base>_<r>_<suffix>")
    ap.add_argument("--out",        required=True, help="Output NetCDF path")
    ap.add_argument("--crs",        default="EPSG:2056", help="CRS to attach to variables (default EPSG:2056)")
    ap.add_argument("--skip-missing", action="store_true",
                    help="Skip missing variable files (e.g., if Qx/Qy not present)")
    args = ap.parse_args()

    parent   = args.parent_dir
    base     = args.base
    r_start  = args.r_start
    r_end    = args.r_end
    t_start  = args.start
    t_end    = args.end
    width    = args.width
    suffix   = args.ens_suffix
    out_path = args.out

    # temp store per-realization NetCDFs
    tmp_dir = tempfile.mkdtemp(prefix="ens_tmp_")
    tmp_paths = []

    try:
        for r in range(r_start, r_end + 1):
            folder = os.path.join(parent, f"{base}_{r}_{suffix}")
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Realization folder not found: {folder}")

            out_r = os.path.join(tmp_dir, f"{base}_{r}_stack.nc")
            print(f"⏳ stacking realization {r} from {folder} -> {out_r}")

            ascii_ensemble_to_netcdf(
                out_nc=out_r,
                folder=folder,
                base=base,
                start=t_start, end=t_end, width=width,
                realization=r,         # add realization dim
                crs=args.crs,
                skip_missing=args.skip_missing,
            )
            tmp_paths.append(out_r)

        print("⏳ concatenating all realizations into one file…")
        ds_list = [xr.open_dataset(p) for p in tmp_paths]
        big = xr.concat(ds_list, dim="realization").sortby("realization")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        big.to_netcdf(out_path)
        for ds in ds_list:
            ds.close()

        print(f"✅ Saved: {out_path}")
    finally:
        # clean temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
