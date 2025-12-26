#!/usr/bin/env -S mamba run -n env_py311 python
import os, sys, argparse
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent          # .../Liestal_event/scripts
SRC_DIR  = (THIS_DIR / ".." / "src").resolve()     # .../Liestal_event/src
sys.path.insert(0, str(SRC_DIR))

print("Using SRC_DIR =", SRC_DIR)

from flow_depth_plotting import lisflood_ensemble_to_forecastlike_netcdf  # noqa: E402


def _sanitize_filename(name: str) -> str:
    # avoid ":" in filenames (causes lots of pain in shells/tools)
    return name.replace(":", "")


def main():
    ap = argparse.ArgumentParser(description="Pack LISFLOOD ASCII ensemble into COSMO-like NetCDF")
    ap.add_argument("--parent-dir", required=True, help="Folder containing realization subfolders")
    ap.add_argument("--base", required=True, help="Base file prefix (e.g. Zell_2m)")
    ap.add_argument("--r-start", type=int, required=True, help="First realization index")
    ap.add_argument("--r-end", type=int, required=True, help="Last realization index")
    ap.add_argument("--start", type=int, required=True, help="First time-step index (e.g. 0)")
    ap.add_argument("--end", type=int, required=True, help="Last time-step index (inclusive)")
    ap.add_argument("--width", type=int, default=4, help="Zero-padding width (default 4 for 0001)")
    ap.add_argument("--ens-suffix", default="fv1-gpu", help="Realization folder suffix, e.g., fv1-gpu")

    # IMPORTANT: let user pass filename or path
    ap.add_argument("--out", required=True, help="Output NetCDF path (filename or full path)")

    # time settings (COSMO-like)
    ap.add_argument("--reference-start", required=True,
                    help="Init time, e.g. 2022-05-05T12:00:00.000000000")
    ap.add_argument("--dt-minutes", type=int, default=60, help="Minutes per model step (default 60)")

    # data settings
    ap.add_argument("--crs", default="EPSG:2056", help="CRS to attach (default EPSG:2056)")
    ap.add_argument("--skip-missing", action="store_true", help="Skip missing files if some vars absent")
    args = ap.parse_args()

    # ---- FORCE output into the directory you want ----
    TARGET_DIR = "/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Zell_2m"

    out_arg = _sanitize_filename(args.out)

    # If user gave only a filename or relative path -> write into TARGET_DIR
    if not os.path.isabs(out_arg):
        out_path = os.path.join(TARGET_DIR, os.path.basename(out_arg))
    else:
        out_path = out_arg

    # Normalize path (absolute, no "..")
    out_path = os.path.abspath(out_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- build realization folders ----
    parent = args.parent_dir
    base = args.base
    suffix = args.ens_suffix

    member_folders = []
    for r in range(args.r_start, args.r_end + 1):
        folder = os.path.join(parent, f"{base}_{r}_{suffix}")
        if not os.path.isdir(folder):
            print(f"⚠️ Missing folder: {folder}")
            continue
        member_folders.append(folder)

    if not member_folders:
        print("⚠️ No realization folders found — nothing to do.")
        return

    # ---- run writer ----
    out_nc = lisflood_ensemble_to_forecastlike_netcdf(
        out_nc=out_path,
        member_folders=member_folders,
        base=base,
        start=args.start,
        end=args.end,
        width=args.width,
        crs=args.crs,
        skip_missing=args.skip_missing,
        reference_start=args.reference_start,
        dt_minutes=args.dt_minutes,
    )

    print(f"✅ Saved successfully: {out_nc}")


if __name__ == "__main__":
    main()

