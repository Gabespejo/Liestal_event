#!/usr/bin/env -S mamba run -n env_py311 python
import sys
import argparse

# Make sure your src/ is on the path
sys.path.insert(0, "/storage/homefs/ge24z347/Zell_event/src/")

from files_preparing_to_lisflood import crop_deterministic_Combiprecip_bounds


def _parse_times_csv(s: str):
    # "2024-06-25T15:00:00,2024-06-25T16:00:00,..."
    return [t.strip() for t in s.split(",") if t.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Crop Combiprecip NetCDF using *_bounds.txt from a DEM/.dem and selected timestamps."
    )
    p.add_argument("--orig-nc", required=True, help="Input CPC NetCDF")
    p.add_argument("--dem-file", required=True,
                   help="DEM file path used to locate *_bounds.txt (can be .tif or .dem)")
    p.add_argument("--output-nc", required=True, help="Output cropped NetCDF")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--times", help="Comma-separated ISO times")
    g.add_argument("--times-file", help="Text file with one ISO time per line")

    # Optional overrides (only needed if your CPC file uses different names)
    p.add_argument("--time-name", default="REFERENCE_TS")
    p.add_argument("--var-name", default="CPC")
    p.add_argument("--x-name", default="x")
    p.add_argument("--y-name", default="y")
    p.add_argument("--snap-res", type=int, default=1000)

    args = p.parse_args()

    if args.times:
        selected_times = _parse_times_csv(args.times)
    else:
        with open(args.times_file) as f:
            selected_times = [ln.strip() for ln in f if ln.strip()]

    crop_deterministic_Combiprecip_bounds(
        orig_nc=args.orig_nc,
        dem_file=args.dem_file,
        output_nc=args.output_nc,
        selected_times=selected_times,
        time_name=args.time_name,
        var_name=args.var_name,
        x_name=args.x_name,
        y_name=args.y_name,
        snap_res=args.snap_res,
    )


if __name__ == "__main__":
    main()
