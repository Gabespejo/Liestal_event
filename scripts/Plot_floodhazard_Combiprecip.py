#!/usr/bin/env -S mamba run -n env_py311 python

import sys
import os
import argparse

# Add src/ to Python path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from flow_depth_plotting import plot_deterministic_perhour


def parse_lead_times(s: str):
    """
    Parse lead times string like "0-12" → [0,1,2,...,12]
    or "0,3,6,12" → [0,3,6,12].
    """
    if "-" in s:
        start, end = map(int, s.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser(
        description="Plot LISFLOOD deterministic water depth maps (no ensembles)"
    )
    p.add_argument("--case-name", required=True,
                   help="Case name, e.g. Zell_2m, Bern_2m")
    p.add_argument("--lead-times", required=True,
                   help="Lead times: either '0-12' or '0,3,6,12'")

    # Option 1: forecast-style start datetime
    p.add_argument("--initial-datetime", required=False,
                   help="Optional start datetime (ISO, e.g. 2024-06-25T15:00:00). "
                        "If omitted and no --forecast-times, titles are 'Hour N'.")

    # Option 2: observational times list
    p.add_argument("--forecast-times", nargs="+", required=False,
                   help="Explicit observational timestamps (ISO, space separated). "
                        "Example: 2022-05-05T08:00:00 2022-05-05T09:00:00 ...")

    # 🎨 Color arguments
    p.add_argument("--color1", default="#fee8c8",
                   help="Hex code or name for Low category (default: #fee8c8)")
    p.add_argument("--color2", default="#fdb366",
                   help="Hex code or name for Medium category (default: #fdb366)")
    p.add_argument("--color3", default="#b30000",
                   help="Hex code or name for High category (default: #b30000)")

    args = p.parse_args()

    case = args.case_name

    # Base build dir (where .dem and _bounds.txt live)
    base_dir = f"/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/{case}"

    # Subfolder with .wd outputs
    wd_dir = os.path.join(base_dir, case)

    dem_file = os.path.join(base_dir, f"{case}.dem")
    bounds_file = os.path.join(base_dir, f"{case}_bounds.txt")

    # read bounding box
    with open(bounds_file, "r") as f:
        minx, miny, maxx, maxy = map(float, f.read().strip().split(","))
    xlim = (minx, maxx)
    ylim = (miny, maxy)

    # parse lead times
    lead_times_hours = parse_lead_times(args.lead_times)

    output_folder = f"/storage/homefs/ge24z347/LISFLOOD_plots/{case}"

    if not os.path.exists(base_dir):
        print(f" Build folder not found: {base_dir}")
        return

    print(f" Plotting deterministic run: {case}")

    # Choose title mode
    forecast_times = args.forecast_times if args.forecast_times else None

    plot_deterministic_perhour(
        case_name=case,
        dem_file=dem_file,
        wd_folder=wd_dir,  # 👈 subfolder with .wd
        plot_output_folder=output_folder,
        lead_times_hours=lead_times_hours,
        initial_datetime_str=args.initial_datetime if args.initial_datetime else None,
        forecast_times=forecast_times,  # 👈 NEW
        color1=args.color1,
        color2=args.color2,
        color3=args.color3,
        xlim=xlim,
        ylim=ylim,
    )

    print(f" Plotting finished for {case}.")


if __name__ == "__main__":
    main()
