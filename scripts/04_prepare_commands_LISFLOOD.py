#!/usr/bin/env -S mamba run -n env_py311 python
import os, sys, argparse

# point src/ at your modules
sys.path.insert(0, os.path.abspath(os.path.join(__file__,"..","..","src")))

from lisflood_inputdata import (
    copy_and_rename_n_file,
    copy_and_rename_dem_file,
    various_stage_files_quadratic,
    various_par_files_Liestal_10hour
)

def main():
    p = argparse.ArgumentParser(
        description="Auto-generate LISFLOOD scenario files (n/.dem/.stage/.par)"
    )
    p.add_argument("--build-dir",   required=True,
                   help="where Liestal_2m.n & .dem live and will write numbered files")
    p.add_argument("--location-id", type=int, default=2,
                   help="ID for your .stage files")
    p.add_argument("--start",       type=int, default=1,
                   help="first scenario index")
    p.add_argument("--end",         type=int, default=11,
                   help="last scenario index (inclusive)")
    p.add_argument("--step",        type=int, default=1,
                   help="step between scenario indices")
    args = p.parse_args()

    bd = args.build_dir
    # 1) copy & rename the .n and .dem into Liestal_2m_1.n ... Liestal_2m_11.n etc.
    copy_and_rename_n_file(
        base_file = os.path.join(bd, "Liestal_2m.n"),
        output_dir= bd,
        start     = args.start,
        end       = args.end,
        step      = args.step
    )
    copy_and_rename_dem_file(
        base_file = os.path.join(bd, "Liestal_2m.dem"),
        output_dir= bd,
        start     = args.start,
        end       = args.end,
        step      = args.step
    )

    # 2) stage files for each scenario
    various_stage_files_quadratic(
        dem_file_path = os.path.join(bd, "Liestal_2m.dem"),
        selected_id   = args.location_id,
        buffer_start  = args.start,
        buffer_end    = args.end,
        buffer_step   = args.step,
        num_points    = 1
    )

    # 3) par files (10-hour) for each scenario
    various_par_files_Liestal_10hour(
        dem_file_path = os.path.join(bd, "Liestal_2m.dem"),
        buffer_start  = args.start,
        buffer_end    = args.end,
        buffer_step   = args.step
    )

    print("✅ All scenarios generated.")

if __name__=="__main__":
    main()