#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 1) Start clean
echo " Purging environment modules..."
module purge

# 2) Load each module explicitly, and confirm
echo " Loading modules..."
module load foss || { echo " Failed to load foss"; exit 1; }
module load CMake || { echo " Failed to load CMake"; exit 1; }
module load netCDF/4.9.2-gompi-2023a || { echo " Failed to load netCDF"; exit 1; }
module load CUDA || { echo " Failed to load CUDA"; exit 1; }

# Activate the Python environment (mamba-based)
echo " Activating env_py311 using mamba..."
source ~/mambaforge/etc/profile.d/conda.sh
mamba activate env_py311

# 3) Show versions
echo " Python: $(which python)  --  $(python --version)"
echo " nvcc: $(which nvcc)  --  $(nvcc --version | head -n1)"

echo " Loaded modules:"
module list 2>&1

# 4) Check GPU access
echo "  Available GPUs:"
nvidia-smi

echo " Environment check completed successfully."

# ─── 3) set up paths ─────────────────────────────────────────────────────────────
BASE=/storage/homefs/ge24z347/LISFLOOD_FP_8_1
SCRIPTS=$BASE/scripts
RAW_CSV=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/Data_forprocess/catchment_location.csv
DEM_SRC=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/Data_forprocess/LIESTAL_DEM_2M
WORK_DEM=$DEM_SRC/LIESTAL
BUILD_DIR=$BASE/build/Liestal_2m

mkdir -p $WORK_DEM $BUILD_DIR logs

# ─── 4) 01_prepare_dem ──────────────────────────────────────────────────────────
echo "→ [1/5] preparing DEM..."
python $SCRIPTS/01_prepare_dem.py \
    --location-csv $RAW_CSV \
    --dem-folder   $DEM_SRC \
    --work-folder  $WORK_DEM \
    --output-dem   $BUILD_DIR/Liestal_2m.dem \
    --location-id  2 \
    --width        2000 \
    --height       2000 \
    --resolution   2.0

# ─── 5) 02_crop_icon ─────────────────────────────────────────────────────────────
echo "→ [2/5] cropping ICON to DEM..."
python $SCRIPTS/02_crop_icon.py \
    --orig-nc        /storage/homefs/ge24z347/LISFLOOD_FP_8_1/Data_forprocess/ICON_Forecast/icon_forecasts_june_24_swiss_radar_grid.nc \
    --dem-file       $BUILD_DIR/Liestal_2m.dem \
    --output-folder  $BUILD_DIR \
    --target-time    2024-06-25T00:00:00 \
    --max-lead-hours 10

# ─── 6) 03_prepare_manning ─────────────────────────────────────────────────────
echo "→ [3/5] generating manning raster..."
python $SCRIPTS/03_prepare_manning.py \
    --geopackage     /rs_scratch/users/ge24z347/Arealstatistik_processing/arealstatistik.gpkg \
    --code-column    LC_27 \
    --output-work    $BUILD_DIR

# ─── 7) 04_create_commands ─────────────────────────────────────────────────────
echo "→ [4/5] creating .stage & .par files..."
python $SCRIPTS/04_create_commands.py \
    --dem-file       $BUILD_DIR/Liestal_2m.dem \
    --n-file         $BUILD_DIR/Liestal_2m.n \
    --nc-folder      $BUILD_DIR \
    --id              2 \
    --start          1 \
    --end            11 \
    --step           1

# ─── 5) run the actual LISFLOOD ensemble ───────────────────────────────────
echo "→ [5/5] run LISFLOOD…"
bash scripts/various_scenarios_lisflood_solvers.sh \
     Liestal_2m 1 11 1 fv1-gpu

echo "✅ All done!"