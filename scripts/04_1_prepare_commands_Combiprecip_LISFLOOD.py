#!/usr/bin/env -S mamba run -n env_py311 python
import os, sys, argparse

# point src/ at your modules
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from DEM_processing import (
    create_stage_file,
    create_par_file_Combiprecip
)

from lisflood_inputdata import write_bci_qflex 

def main():
    p = argparse.ArgumentParser(
        description="Auto-generate LISFLOOD deterministic scenario files (n/.dem/.stage/.par/.bci)"
    )
    p.add_argument("--build-dir",    required=True,
                   help="Folder where base .n, .dem, .nc live and outputs will be written")
    p.add_argument("--base-name",    required=True,
                   help="Base name (e.g., Zell_2m)")
    p.add_argument("--catchment-csv", required=True,
                   help="CSV with catchment locations (must contain ID, East_X, North_Y)")
    p.add_argument("--location-id",  type=int, required=True,
                   help="ID for .stage file (from CSV)")
    p.add_argument("--time",         type=int, required=True,
                   help="Simulation time in seconds (e.g. 46800 for 13h)")

    # inflow/outflow options
    p.add_argument("--q-m3s",        type=float,
                   help="Total inflow discharge in m³/s")
    p.add_argument("--cell-size",    type=float,
                   help="Grid cell size in m (required if using point inflows)")
    p.add_argument("--point-inflow", nargs=2, action="append", metavar=("X", "Y"),
                   help="Add a point inflow coordinate (can be repeated)")
    p.add_argument("--line-inflow", nargs=3, action="append", metavar=("SIDE","START","END"),
                   help="Add a line inflow segment (can be repeated)")
    p.add_argument("--outflow-side", default="E",
                   help="Side for FREE outflow (default: E)")
    p.add_argument("--outflow-start", type=float, required=True,
                   help="Start coordinate for FREE outflow")
    p.add_argument("--outflow-end",   type=float, required=True,
                   help="End coordinate for FREE outflow")
    p.add_argument("--outflow-slope", type=float,
                   help="Optional slope after FREE")

    args = p.parse_args()

    base = args.base_name
    bd   = args.build_dir

    # 1) Create .stage file
    create_stage_file(
        catchment_location_csv=args.catchment_csv,
        selected_id=args.location_id,
        output_stage_file=os.path.join(bd, f"{base}.stage"),
        num_points=1
    )

    # 2) Create .par file
    create_par_file_Combiprecip(
        base_name=base,
        time=args.time,
        output_file_path=os.path.join(bd, f"{base}.par")
    )

    # 3) Create .bci file if inflow/outflow arguments are provided
    bci_path = os.path.join(bd, f"{base}.bci")

    point_inflows = [(float(x), float(y)) for (x, y) in (args.point_inflow or [])]
    line_inflows = []
    if args.line_inflow:
        for side, start, end in args.line_inflow:
            line_inflows.append({"side": side, "start": float(start), "end": float(end)})

    write_bci_qflex(
        output_path=bci_path,
        Q_m3s=args.q_m3s,
        cell_size=args.cell_size,
        point_inflows=point_inflows,
        line_inflows=line_inflows,
        outflow_side=args.outflow_side,
        outflow_start=args.outflow_start,
        outflow_end=args.outflow_end,
        outflow_slope=args.outflow_slope,
    )

if __name__ == "__main__":
    main()