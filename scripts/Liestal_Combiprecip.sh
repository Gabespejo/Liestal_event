#!/bin/bash
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preemptable
#SBATCH --time=05:00:00
#SBATCH --output=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/scripts/logs/%x_%j.out
#SBATCH --error=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/scripts7logs/%x_%j.err

# 1) Start clean
echo "Purging environment modules..."
module purge

# 2) Load modules
echo "Loading modules..."
module load foss || { echo "Failed to load foss"; exit 1; }
module load CMake || { echo "Failed to load CMake"; exit 1; }
module load netCDF/4.9.2-gompi-2023a || { echo "Failed to load netCDF"; exit 1; }
module load CUDA || { echo "Failed to load CUDA"; exit 1; }

# 3) Show module versions
echo "nvcc: $(which nvcc)  --  $(nvcc --version | head -n1)"
module list 2>&1

# 4) Check GPU access
echo "Available GPUs:"
nvidia-smi

# 5) Activate conda environment
echo "Activating conda environment env_py311..."
source /storage/homefs/ge24z347/mambaforge/etc/profile.d/conda.sh
conda activate env_py311 || { echo "Failed to activate environment"; exit 1; }

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
sleep 5  # Delay to let CUDA init cleanly
monitor_gpu_usage &
MONITOR_PID=$!

# Trap to kill monitor on exit or failure
trap "echo 'Killing GPU monitoring...'; kill $MONITOR_PID" EXIT

# 7) Run LISFLOOD simulation
echo "Running LISFLOOD simulation..."
cd /storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Liestal_2m/ || { echo "Failed to cd into simulation directory"; exit 1; }
/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/lisflood /storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Liestal_2m/Liestal_2m.par

# 8) Done
echo "LISFLOOD simulation completed."