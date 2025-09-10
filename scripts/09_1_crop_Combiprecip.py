#!/usr/bin/env -S mamba run -n env_py311 python
import sys, argparse

# Make sure your src/ is on the path
sys.path.insert(0, "/storage/homefs/ge24z347/Liestal_event/src/")

from lisflood_inputdata import crop_deterministic_Combiprecip_bounds


def _parse_times_csv(s: str):
    # "2024-06-25T15:00:00,2024-06-25T16:00:00,..."
    return [t.strip() for t in s.split(",") if t.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Crop Combiprecip NetCDF to DEM bounds (from *_bounds.txt) and selected timestamps."
    )
    p.add_argument("--orig-nc",    required=True, help="Input CPC NetCDF")
    p.add_argument("--dem-file",   required=True,
                   help="DEM file (only used to find *_bounds.txt in the same folder)")
    p.add_argument("--output-nc",  required=True, help="Output cropped NetCDF")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--times",      help="Comma-separated ISO times")
    g.add_argument("--times-file", help="Text file with one ISO time per line")

    args = p.parse_args()

    # Collect times
    if args.times:
        selected_times = _parse_times_csv(args.times)
    else:
        with open(args.times_file) as f:
            selected_times = [ln.strip() for ln in f if ln.strip()]

    # Call your function
    crop_deterministic_Combiprecip_bounds(
        orig_nc=args.orig_nc,
        dem_file=args.dem_file,
        output_nc=args.output_nc,
        selected_times=selected_times,
        # If needed, you can override CPC variable names with
        # time_name="REFERENCE_TS", var_name="CPC", etc.
    )


if __name__ == "__main__":
    main()