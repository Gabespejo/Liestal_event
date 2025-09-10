#!/usr/bin/env -S mamba run -n env_py311 python
import os, sys, argparse

# point src/ at your modules
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from DEM_processing import (
    create_stage_file,
    create_par_file_Combiprecip_bdy_dg2
)
from lisflood_inputdata import write_bci_qvar, write_bdy_qvar


def main():
    p = argparse.ArgumentParser(description="Generate LISFLOOD input files (.par, .bci, .bdy, .stage)")
    subparsers = p.add_subparsers(dest="command", required=True)

    # ---------- PAR ----------
    par_p = subparsers.add_parser("par", help="Generate .par file")
    par_p.add_argument("--base-name", required=True, help="Base name (e.g. Zell_2m_bach)")
    par_p.add_argument("--sim-time", type=int, required=True, help="Simulation time in seconds")

    # ---------- BCI ----------
    bci_p = subparsers.add_parser("bci", help="Generate .bci file (QVAR inflows)")
    bci_p.add_argument("--base-name", required=True, help="Base name (e.g. Zell_2m_bach)")
    bci_p.add_argument("--outflow-side", required=True, help="FREE outflow side (N,S,E,W)")
    bci_p.add_argument("--outflow-start", type=float, required=True, help="Outflow start coordinate")
    bci_p.add_argument("--outflow-end", type=float, required=True, help="Outflow end coordinate")
    bci_p.add_argument("--inflow", nargs=3, action="append", metavar=("NAME", "X", "Y"),
                       help="Add inflow point: NAME X Y (repeat as needed)")

    # ---------- BDY ----------
    bdy_p = subparsers.add_parser("bdy", help="Generate .bdy file (QVAR time series)")
    bdy_p.add_argument("--base-name", required=True, help="Base name (e.g. Zell_2m_bach)")
    bdy_p.add_argument("--times", nargs="+", type=int, required=True, help="List of time steps in seconds")
    bdy_p.add_argument(
        "--inflow",
        nargs="+",
        action="append",
        metavar="ARGS",
        help="Inflow series: NAME WIDTH Q1 Q2 ... Qn"
    )

    args = p.parse_args()

    # build folder path
    build_dir = f"/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/{args.base_name}"
    os.makedirs(build_dir, exist_ok=True)

    # ---------- RUN ----------
    if args.command == "par":
        output_path = os.path.join(build_dir, f"{args.base_name}.par")
        create_par_file_Combiprecip_bdy_dg2(
            base_name=args.base_name,
            time=args.sim_time,
            output_file_path=output_path
        )
        print(f"✔ PAR file written → {output_path}")

    elif args.command == "bci":
        inflows = {}
        for name, x, y in args.inflow:
            inflows.setdefault(name, []).append((float(x), float(y)))

        # Save inside build folder
        output_path = os.path.join(build_dir, f"{args.base_name}.bci")

        lines = []
        # Preserve order of appearance (not sorted)
        for name in inflows:
            for x, y in inflows[name]:
                lines.append(f"P {x:.3f} {y:.3f} QVAR {name}")

        # Outflow
        lines.append(f"{args.outflow_side} {args.outflow_start:.3f} {args.outflow_end:.3f} FREE")

        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"✔ BCI file written → {output_path}")

    elif args.command == "bdy":
        inflows = {}
        n_times = len(args.times)

        for inflow in args.inflow:
            name = inflow[0]
            width = float(inflow[1])
            discharges = [float(v) for v in inflow[2:] if v.strip() != ""]

            if len(discharges) != n_times:
                raise ValueError(
                    f"Inflow {name} has {len(discharges)} discharges but {n_times} times were provided."
                )

            inflows[name] = {"width": width, "discharges": discharges}

        output_path = os.path.join(build_dir, f"{args.base_name}.bdy")
        write_bdy_qvar(
            output_path=output_path,
            inflows=inflows,
            times=args.times
        )
        print(f"✔ BDY file written → {output_path}")


if __name__ == "__main__":
    main()