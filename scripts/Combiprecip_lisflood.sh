#!/bin/bash
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preemptable
#SBATCH --time=15:00:00
#SBATCH --output=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/scripts/logs/%x_%j.out
#SBATCH --error=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/scripts/logs/%x_%j.err

# Usage: sbatch Liestal_Combiprecip.sh Zell_2m
#        sbatch Liestal_Combiprecip.sh Buochs_2m_v6

# 0) Get case name from input
CASE_NAME="$1"
if [ -z "$CASE_NAME" ]; then
  echo "❌ Error: You must provide a case name (e.g. sbatch Liestal_Combiprecip.sh Zell_2m)"
  exit 1
fi

BUILD_DIR="/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/${CASE_NAME}"
PAR_FILE="${BUILD_DIR}/${CASE_NAME}.par"
EXEC="/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/lisflood"

# 1) Start clean
echo "Purging environment modules..."
module purge

# 2) Load modules
echo "Loading modules..."
module load foss || exit 1
module load CMake || exit 1
module load netCDF/4.9.2-gompi-2023a || exit 1
module load CUDA || exit 1

# 3) Show module versions
echo "nvcc: $(which nvcc)  --  $(nvcc --version | head -n1)"
module list 2>&1

# 4) Check GPU access
echo "Available GPUs:"
nvidia-smi

# 5) Activate conda environment
echo "Activating conda environment env_py311..."
source /storage/homefs/ge24z347/mambaforge/etc/profile.d/conda.sh
conda activate env_py311 || exit 1

# 6) Start GPU monitoring safely
GPU_LOG="/storage/homefs/ge24z347/gpu_usage.log"
echo "Starting GPU monitoring in background..."
monitor_gpu_usage() {
    while true; do
        nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total \
        --format=csv,noheader >> "$GPU_LOG"
        sleep 60
    done
}
sleep 5
monitor_gpu_usage &
MONITOR_PID=$!
trap "kill $MONITOR_PID" EXIT

# 7) Run LISFLOOD simulation
echo "Running LISFLOOD simulation for case ${CASE_NAME}..."
cd "$BUILD_DIR" || { echo "Failed to cd into $BUILD_DIR"; exit 1; }
"$EXEC" "$PAR_FILE"

# 8) Done
echo "LISFLOOD simulation completed for ${CASE_NAME}."