#!/usr/bin/env -S mamba run -n env_py311 python
import argparse
from pathlib import Path
import sys

# make sure Python can find your module if it's in src/
sys.path.append("/storage/homefs/ge24z347/Zell_event/src")

# ✅ import the correct unstructured plotting function
from flow_depth_plotting import plot_h_unstructured_colored_edges


def parse_extent(s: str):
    """
    Parse "xmin,xmax,ymin,ymax" into a 4-tuple of floats.
    """
    try:
        parts = [float(p.strip()) for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError
        return tuple(parts)
    except Exception:
        raise argparse.ArgumentTypeError("extent must be 'xmin,xmax,ymin,ymax' (comma-separated numbers)")


def main():
    p = argparse.ArgumentParser(
        description="Plot unstructured UGRID h polygons with swisstopo background + feature edges."
    )
    p.add_argument("--nc-path", required=True, help="Path to unstructured NetCDF (UGRID)")
    p.add_argument("--out-dir", required=True, help="Folder to save PNGs")
    p.add_argument("--threshold", type=float, default=0.01, help="Mask threshold (m): values < threshold are transparent")
    p.add_argument("--vmax", type=float, default=3.5, help="Max for class bins (m)")
    p.add_argument("--case-label", default="Zell", help="Title prefix / case label")
    p.add_argument("--init-time", default=None, help="Init time string used in filename/title (optional)")
    p.add_argument("--extent", type=parse_extent, default=None,
                   help="Plot extent as 'xmin,xmax,ymin,ymax' (in extent-units)")
    p.add_argument("--extent-units", choices=["m", "km"], default="m",
                   help="Units for --extent values")
    p.add_argument("--bg-pixel-size", type=float, default=1.0,
                   help="Background WMS pixel size (m)")
    p.add_argument("--bg-max-px", type=int, default=4096,
                   help="Background WMS maximum dimension (px)")
    p.add_argument("--layer", default="ch.swisstopo.swisstlm3d-karte-grau",
                   help="Swisstopo WMS layer id")
    p.add_argument("--linewidth", type=float, default=0.12,
                   help="Feature edge linewidth")
    p.add_argument("--dpi", type=int, default=200,
                   help="Figure DPI")

    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ✅ Call the unstructured plotting function
    plot_h_unstructured_colored_edges(
        nc_path=args.nc_path,
        out_dir=args.out_dir,
        threshold=args.threshold,
        vmax=args.vmax,
        layer=args.layer,
        bg_pixel_size=args.bg_pixel_size,
        bg_max_px=args.bg_max_px,
        case_label=args.case_label,
        init_time_str=args.init_time,
        extent=args.extent,
        extent_units=args.extent_units,
        linewidth=args.linewidth,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
