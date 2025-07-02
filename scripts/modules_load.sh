#!/bin/bash
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preemptable 
#SBATCH --time=00:04:00
#SBATCH --output=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/scripts/logs/%x_%j.out
#SBATCH --error=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/scripts7logs/%x_%j.err

# 1) Start clean
echo " Purging environment modules..."
module purge

# 2) Load each module explicitly, and confirm
echo " Loading modules..."
module load foss || { echo " Failed to load foss"; exit 1; }
module load CMake || { echo " Failed to load CMake"; exit 1; }
module load netCDF/4.9.2-gompi-2023a || { echo " Failed to load netCDF"; exit 1; }
module load CUDA || { echo " Failed to load CUDA"; exit 1; }

# 3) Show versions
echo " nvcc: $(which nvcc)  --  $(nvcc --version | head -n1)"

echo " Loaded modules:"
module list 2>&1

# 4) Check GPU access
echo "  Available GPUs:"
nvidia-smi

echo " Environment check completed successfully."


source /storage/homefs/ge24z347/mambaforge/etc/profile.d/conda.sh
conda activate env_py311



