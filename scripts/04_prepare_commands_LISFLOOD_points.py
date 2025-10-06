#!/usr/bin/env -S mamba run -n env_py311 python
import os, sys, argparse, re, shutil

# make sure src/ is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from DEM_processing import (
    create_par_file_Forecast_bdy_qoutput  # writes a .par (expects base_name, time, cfl, output_file_path)
)
from lisflood_inputdata import write_bci_qvar, write_bdy_qvar


def strip_bdy_from_par(par_path: str) -> None:
    """Remove any line that references a BDY file or time-varying BC."""
    if not os.path.exists(par_path):
        return
    with open(par_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    patterns = [
        r"\bBDY\b", r"\bBDYFILE\b", r"\.bdy\b", r"\.BDY\b",
        r"\bQVAR\b", r"\bTIME_SERIES\b", r"\bTIMESERIES\b"
    ]
    rx = re.compile("|".join(patterns), flags=re.IGNORECASE)
    kept = [ln for ln in lines if not rx.search(ln)]

    with open(par_path, "w", encoding="utf-8") as f:
        f.writelines(kept)


def set_par_key(par_path: str, key: str, value: str) -> None:
    """
    In-place replace a single-line key with a new value using LISFLOOD's fixed column format:
    f"{key:25} {value}\n". If key is missing, append it.
    """
    key_lower = key.strip().lower()
    new_line = f"{key:25} {value}\n"
    out_lines = []
    replaced = False
    with open(par_path, "r", encoding="utf-8") as f:
        for ln in f:
            # match beginning of line ignoring spaces and case
            if ln.strip().lower().startswith(key_lower):
                out_lines.append(new_line)
                replaced = True
            else:
                out_lines.append(ln)
    if not replaced:
        out_lines.append(new_line)
    with open(par_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)


def main():
    p = argparse.ArgumentParser(
        description="Generate LISFLOOD input files (.par, .bci, .bdy, .stage) — with optional replicated filenames"
    )
    subparsers = p.add_subparsers(dest="command", required=True)

    # ---------- PAR ----------
    par_p = subparsers.add_parser("par", help="Generate .par file (optionally replicated as base_1.par..base_N.par)")
    par_p.add_argument("--base-name", required=True, help="Base name (e.g. Zell_2m)")
    par_p.add_argument("--sim-time", type=int, required=True, help="Simulation time in seconds")
    par_p.add_argument("--cfl", type=float, default=0.25, help="CFL number (default: 0.25)")
    par_p.add_argument("--ens-count", type=int, default=1,
                       help="How many copies to write as base_1.par..base_N.par (default: 1)")
    par_p.add_argument(
        "--add-bdy", action="store_true",
        help="Include a .bdy reference in the .par (use when you have QVAR inflows). "
             "If omitted, the .par will reference only the .bci."
    )

    # ---------- BCI ----------
    bci_p = subparsers.add_parser("bci", help="Generate .bci file (optionally replicated as base_1.bci..base_N.bci)")
    bci_p.add_argument("--base-name", required=True, help="Base name (e.g. Zell_2m)")
    bci_p.add_argument("--outflow-side", required=True, help="FREE outflow side (N,S,E,W)")
    bci_p.add_argument("--outflow-start", type=float, required=True, help="Outflow start coordinate")
    bci_p.add_argument("--outflow-end", type=float, required=True, help="Outflow end coordinate")
    bci_p.add_argument("--ens-count", type=int, default=1,
                       help="How many copies to write as base_1.bci..base_N.bci (default: 1)")
    bci_p.add_argument(
        "--inflow", nargs=3, action="append", metavar=("NAME", "X", "Y"),
        help="Add inflow point: NAME X Y (repeat as needed). Optional."
    )
    bci_p.add_argument(
        "--outflow-point-free", action="append", metavar="X,Y",
        help="Add FREE outflow point as 'X,Y'. Repeat for multiple points."
    )

    # ---------- BDY ----------
    bdy_p = subparsers.add_parser("bdy", help="Generate .bdy file (single file)")
    bdy_p.add_argument("--base-name", required=True, help="Base name (e.g. Zell_2m)")
    bdy_p.add_argument("--times", nargs="+", type=int, required=True, help="List of time steps in seconds")
    bdy_p.add_argument(
        "--inflow", nargs="+", action="append", metavar="ARGS", required=True,
        help="Inflow series: NAME WIDTH Q1 Q2 ... Qn (repeat per inflow)"
    )

    # ---------- STAGE ----------
    stage_p = subparsers.add_parser("stage", help="Generate .stage file (optionally replicated as base_1.stage..base_N.stage)")
    stage_p.add_argument("--base-name", required=True, help="Base name (e.g. Zell_2m)")
    stage_p.add_argument("--ens-count", type=int, default=1,
                         help="How many copies to write as base_1.stage..base_N.stage (default: 1)")
    stage_p.add_argument(
        "--point", nargs=2, action="append", metavar=("X", "Y"), required=True,
        help="Gauge point coordinates (X Y). Repeat as needed."
    )

    args = p.parse_args()

    # build folder path
    build_dir = f"/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/{args.base_name}"
    os.makedirs(build_dir, exist_ok=True)

    # ---------- RUN ----------
    if args.command == "par":
        # write a base .par with the requested content
        base_par = os.path.join(build_dir, f"{args.base_name}.par")

        create_par_file_Forecast_bdy_qoutput(
            base_name=args.base_name,
            time=args.sim_time,
            cfl=args.cfl,
            output_file_path=base_par
        )

        if not args.add_bdy:
            strip_bdy_from_par(base_par)
            print(f"✔ PAR file written (BCI only — no .bdy) → {base_par}")
        else:
            print(f"✔ PAR file written (includes .bdy) → {base_par}")

        # replicate to base_1.par .. base_N.par and fix lines in each copy
        N = max(1, int(args.ens_count))
        for i in range(1, N + 1):
            par_i = os.path.join(build_dir, f"{args.base_name}_{i}.par")
            shutil.copyfile(base_par, par_i)
            # set dynamicrainfile to <base>_<i>.nc and dirroot to <base>_<i>
            set_par_key(par_i, "dynamicrainfile", f"{args.base_name}_{i}.nc")
            set_par_key(par_i, "dirroot", f"{args.base_name}_{i}")

        if N > 1:
            print(f"✔ Replicated to {N} files with per-copy dynamicrainfile & dirroot:")
            print(f"   {args.base_name}_1.par → dynamicrainfile {args.base_name}_1.nc , dirroot {args.base_name}_1")
            print(f"   …")
            print(f"   {args.base_name}_{N}.par → dynamicrainfile {args.base_name}_{N}.nc , dirroot {args.base_name}_{N}")

    elif args.command == "bci":
        # gather inflows (optional)
        inflows = {}
        if args.inflow:
            for name, x, y in args.inflow:
                inflows.setdefault(name, []).append((float(x), float(y)))

        # gather FREE outflow points (optional)
        outflow_points = []
        if args.outflow_point_free:
            for s in args.outflow_point_free:
                x_str, y_str = s.split(",")
                outflow_points.append((float(x_str), float(y_str)))

        # build content once to a temp file
        base_bci = os.path.join(build_dir, f"{args.base_name}.bci")
        write_bci_qvar(
            output_path=base_bci,
            inflows=inflows if inflows else None,
            outflow_side=args.outflow_side,
            outflow_start=args.outflow_start,
            outflow_end=args.outflow_end,
            outflow_slope=None,
            outflow_points_free=outflow_points if outflow_points else None
        )
        if not inflows and not outflow_points:
            print("ℹ No inflows or FREE points provided. This BCI contains only one FREE outflow segment.")

        # replicate to base_1.bci .. base_N.bci (identical content by design)
        N = max(1, int(args.ens_count))
        for i in range(1, N + 1):
            bci_i = os.path.join(build_dir, f"{args.base_name}_{i}.bci")
            shutil.copyfile(base_bci, bci_i)
        if N > 1:
            print(f"✔ Replicated to {N} files: {args.base_name}_1.bci … {args.base_name}_{N}.bci")

    elif args.command == "bdy":
        inflows = {}
        n_times = len(args.times)

        for inflow in args.inflow:
            name = inflow[0]
            width = float(inflow[1])
            discharges = [float(v) for v in inflow[2:] if str(v).strip() != ""]
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
        print("ℹ Use `--add-bdy` when generating the PAR if you want this BDY to be used.")

    elif args.command == "stage":
        # Gauge stage file: first line is count, then X Y per line
        pts = [(float(x), float(y)) for x, y in args.point]
        base_stage = os.path.join(build_dir, f"{args.base_name}.stage")
        with open(base_stage, "w", encoding="utf-8") as f:
            f.write(f"{len(pts)}\n")
            for x, y in pts:
                f.write(f"{x:.3f} {y:.3f}\n")
        print(f"✔ STAGE (gauge) file written → {base_stage}")

        # replicate to base_1.stage .. base_N.stage (identical content)
        N = max(1, int(args.ens_count))
        for i in range(1, N + 1):
            stage_i = os.path.join(build_dir, f"{args.base_name}_{i}.stage")
            shutil.copyfile(base_stage, stage_i)
        if N > 1:
            print(f"✔ Replicated to {N} files: {args.base_name}_1.stage … {args.base_name}_{N}.stage")

if __name__ == "__main__":
    main()
