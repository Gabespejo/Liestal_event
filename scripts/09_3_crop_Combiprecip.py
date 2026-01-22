#!/usr/bin/env -S mamba run -n env_py311 python
import sys, argparse

# Make sure your src/ is on the path
sys.path.insert(0, "/storage/homefs/ge24z347/Zell_event/src/")

from lisflood_inputdata import crop_deterministic_Combiprecip_inf


def _parse_times_csv(s: str):
    # "2024-06-25T15:00:00,2024-06-25T16:00:00,..."
    return [t.strip() for t in s.split(",") if t.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Crop Combiprecip NetCDF to a DEM footprint and selected timestamps."
    )
    p.add_argument("--orig-nc",    required=True, help="Input CPC NetCDF")
    p.add_argument("--dem-file",   required=True, help="DEM GeoTIFF to get bbox from")
    p.add_argument("--output-nc",  required=True, help="Output cropped NetCDF")

    #  NEW: rainfall multiplicative factor
    p.add_argument(
        "--rain-factor",
        type=float,
        default=1.0,
        help="Multiplicative factor applied to rainfall depth (e.g. 0.9, 1.1)"
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--times",      help="Comma-separated ISO times")
    g.add_argument("--times-file", help="Text file with one ISO time per line")

    args = p.parse_args()

    if args.times:
        selected_times = _parse_times_csv(args.times)
    else:
        with open(args.times_file) as f:
            selected_times = [ln.strip() for ln in f if ln.strip()]

    crop_deterministic_Combiprecip_inf(
        orig_nc=args.orig_nc,
        dem_file=args.dem_file,
        output_nc=args.output_nc,
        selected_times=selected_times,
        rain_factor=args.rain_factor,  #  PASS FACTOR HERE
    )


if __name__ == "__main__":
    main()
