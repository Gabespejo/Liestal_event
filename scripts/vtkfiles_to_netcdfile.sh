#!/bin/bash -l
#SBATCH --account=gratis
#SBATCH --partition=cpu-invest
#SBATCH --qos=job_cpu_preemptable
#SBATCH --job-name=vtk2nc
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/vtk_nc_scripts/%x_%j.out
#SBATCH --error=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/vtk_nc_scripts/%x_%j.err

set -euo pipefail

# ------------------------------------------------------------
# Args (DEM optional)
# ------------------------------------------------------------
if [ "$#" -ne 6 ] && [ "$#" -ne 7 ]; then
  echo "Usage (DEM optional):"
  echo "  With DEM:"
  echo "    sbatch $0 <input_vtk_dir> <output_nc_file> <dem_path> <start_time> <freq> <dx> <epsg>"
  echo ""
  echo "  Without DEM (use full VTK bounds):"
  echo "    sbatch $0 <input_vtk_dir> <output_nc_file> <start_time> <freq> <dx> <epsg>"
  echo ""
  echo "Examples:"
  echo "  sbatch $0 /path/vtks out.nc /path/domain.dem 2022-05-05T12:00:00 1h 0.5 2056"
  echo "  sbatch $0 /path/vtks out.nc 2022-05-05T12:00:00 1h 0.5 2056"
  exit 1
fi

IN_DIR="$1"
OUT_NC="$2"

if [ "$#" -eq 7 ]; then
  DEM_PATH="$3"
  START_TIME="$4"
  FREQ="$5"
  DX="$6"
  EPSG="$7"
else
  DEM_PATH="NONE"   # <-- tell python script to use full VTK bounds
  START_TIME="$3"
  FREQ="$4"
  DX="$5"
  EPSG="$6"
fi

echo "Input VTK dir : $IN_DIR"
echo "Output NetCDF : $OUT_NC"
echo "DEM path      : $DEM_PATH"
echo "Start time    : $START_TIME"
echo "Freq          : $FREQ"
echo "DX            : $DX"
echo "EPSG          : $EPSG"

# ------------------------------------------------------------
# Modules (needed for pvpython!)
# ------------------------------------------------------------
module purge
module load ParaView/5.11.2-foss-2023a

echo "pvpython: $(which pvpython || true)"

# ------------------------------------------------------------
# VTK stability settings (set BEFORE running python)
# ------------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VTK_SMP_IMPLEMENTATION=Sequential
export VTKM_DEVICE_ADAPTER=Serial
export VTKM_NUM_THREADS=1

# ------------------------------------------------------------
# Conda env
# ------------------------------------------------------------
echo "Activating conda env: vtktonetcdfile_py311"
set +u
source /storage/homefs/ge24z347/mambaforge/etc/profile.d/conda.sh
conda activate vtktonetcdfile_py311
set -u

export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
unset PYTHONPATH
export PYTHONNOUSERSITE=1

echo "python: $(which python)"
python -V

echo "Test imports:"
python -c "import pandas as pd, xarray as xr; print('pandas', pd.__version__); print('xarray', xr.__version__)"

# ------------------------------------------------------------
# Run conversion
# ------------------------------------------------------------
PY_SCRIPT="/storage/homefs/ge24z347/Zell_event/scripts/vtkfiles_to_netcdfile.py"

echo "Running conversion..."
python "$PY_SCRIPT" "$IN_DIR" "$OUT_NC" "$DEM_PATH" "$START_TIME" "$FREQ" "$DX" "$EPSG"

echo "Done."
