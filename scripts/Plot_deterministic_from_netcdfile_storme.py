#!/usr/bin/env -S mamba run -n env_py311 python
import argparse
from pathlib import Path
import sys

sys.path.append("/storage/homefs/ge24z347/Zell_event/src")

from flow_depth_plotting import plot_water_depths_deterministic_from_netcdfile_storme


def parse_extent(s: str):
    try:
        parts = [float(p) for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError
        return tuple(parts)
    except Exception:
        raise argparse.ArgumentTypeError(
            "extent must be 'xmin,xmax,ymin,ymax' (comma-separated numbers)"
        )


def main():
    p = argparse.ArgumentParser(
        description="Plot deterministic water-depth PNGs from a NetCDF with optional STORME overlays."
    )

    p.add_argument("--nc-path", required=True, help="Path to NetCDF file")
    p.add_argument("--out-dir", required=True, help="Folder to save PNGs")

    p.add_argument("--threshold", type=float, default=0.10,
                   help="Depth mask threshold (m). Allowed: 0.01 or 0.10")
    p.add_argument("--vmax", type=float, default=3.5,
                   help="Kept for compatibility")
    p.add_argument("--case-label", default="Zell",
                   help="Title prefix / case label")
    p.add_argument("--init-time", default=None,
                   help="Init time string used in title")
    p.add_argument("--extent", type=parse_extent, default=None,
                   help="Plot extent as 'xmin,xmax,ymin,ymax' (in extent-units)")
    p.add_argument("--extent-units", choices=["m", "km", "auto"], default="auto",
                   help="Units for --extent values")
    p.add_argument("--bg-pixel-size", type=float, default=1.0,
                   help="Background WMS pixel size (m)")
    p.add_argument("--bg-max-px", type=int, default=4096,
                   help="Background WMS maximum dimension (px)")
    p.add_argument("--layer", default="ch.swisstopo.swisstlm3d-karte-grau",
                   help="Swisstopo WMS layer id")
    p.add_argument("--min-fig-height", type=float, default=6.0,
                   help="Minimum figure height (inches)")

    p.add_argument("--storm-gpkg", default=None,
                   help="Path to storme_agno_ti_2022.gpkg")
    p.add_argument("--layer-surface-runoff",
                   default="prozessraum_wasser_oberflaechenabfluss_grundwasseraufstoss",
                   help="Surface runoff / groundwater upwelling layer")
    p.add_argument("--layer-flooding",
                   default="prozessraum_wasser_ueberschwemmung_uebermurung",
                   help="Flooding / overbank deposition layer")
    p.add_argument("--color-surface-runoff", default="magenta",
                   help="Color for surface runoff layer")
    p.add_argument("--color-flooding", default="orange",
                   help="Color for flooding layer")
    p.add_argument("--alpha-surface-runoff", type=float, default=0.35,
                   help="Transparency for surface runoff polygons")
    p.add_argument("--alpha-flooding", type=float, default=0.35,
                   help="Transparency for flooding polygons")

    # mask arguments
    p.add_argument("--mask-gpkg",
                   default="/storage/homefs/ge24z347/Zell_event/Data_forprocess/SWISSTLM3D_2025.gpkg",
                   help="Path to SWISSTLM3D GeoPackage used for masking standing water")
    p.add_argument("--mask-layer", default="tlm_bb_bodenbedeckung",
                   help="Layer name inside the mask GeoPackage")
    p.add_argument("--mask-objektart-col", default="objektart",
                   help="Column used to identify standing water polygons")
    p.add_argument("--mask-objektart-value", default="Stehende Gewaesser",
                   help="Value used to select standing water polygons")

    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    plot_water_depths_deterministic_from_netcdfile_storme(
        nc_path=args.nc_path,
        out_dir=args.out_dir,
        threshold=args.threshold,
        vmax=args.vmax,
        bg_pixel_size=args.bg_pixel_size,
        bg_max_px=args.bg_max_px,
        layer=args.layer,
        case_label=args.case_label,
        init_time_str=args.init_time,
        extent=args.extent,
        extent_units=args.extent_units,
        min_fig_height=args.min_fig_height,
        storm_gpkg=args.storm_gpkg,
        layer_surface_runoff=args.layer_surface_runoff,
        layer_flooding=args.layer_flooding,
        color_surface_runoff=args.color_surface_runoff,
        color_flooding=args.color_flooding,
        alpha_surface_runoff=args.alpha_surface_runoff,
        alpha_flooding=args.alpha_flooding,
        mask_gpkg=args.mask_gpkg,
        mask_layer=args.mask_layer,
        mask_objektart_col=args.mask_objektart_col,
        mask_objektart_value=args.mask_objektart_value,
    )


if __name__ == "__main__":
    main()