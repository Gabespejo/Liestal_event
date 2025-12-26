#!/usr/bin/env -S mamba run -n env_py311 python
import argparse
from pathlib import Path
import sys

# make sure Python can find your module if it's in src/
sys.path.insert(0, "/storage/homefs/ge24z347/Liestal_event/src")

from flow_depth_plotting import plot_waterdepth_forecast_from_netcdfile


def parse_extent(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("extent must be 'xmin,xmax,ymin,ymax'")
    try:
        return tuple(float(p) for p in parts)  # (xmin, xmax, ymin, ymax)
    except ValueError:
        raise argparse.ArgumentTypeError("extent values must be numbers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--case-label", default="Zell")
    ap.add_argument("--extent", type=parse_extent, default=None,
                    help="xmin,xmax,ymin,ymax (comma-separated)")
    ap.add_argument("--extent-units", default="m", choices=["m", "km", "auto"])
    ap.add_argument("--min-fig-height", type=float, default=6.0)

    # optional WMS settings if you want to override
    ap.add_argument("--layer", default="ch.swisstopo.swisstlm3d-karte-grau")
    ap.add_argument("--bg-pixel-size", type=float, default=1.0)
    ap.add_argument("--bg-max-px", type=int, default=4096)

    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    plot_waterdepth_forecast_from_netcdfile(
        nc_path=args.nc_path,
        out_dir=args.out_dir,
        threshold=args.threshold,
        case_label=args.case_label,
        extent=args.extent,
        extent_units=args.extent_units,
        min_fig_height=args.min_fig_height,
        layer=args.layer,
        bg_pixel_size=args.bg_pixel_size,
        bg_max_px=args.bg_max_px,
    )


if __name__ == "__main__":
    main()

