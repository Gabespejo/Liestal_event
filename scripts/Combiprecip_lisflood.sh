#!/bin/bash
#SBATCH --job-name=LISFLOOD_Combiprecip
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --time=15:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G

#SBATCH --output=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/scripts/logs/%x_%j.out
#SBATCH --error=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/scripts/logs/%x_%j.err

## ---- EMAIL NOTIFICATION ----
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gabriela.espejogutierrez@unibe.ch

echo "========== SLURM job started at $(date) =========="

# -----------------------------
# 0) Read case name
# -----------------------------
CASE_NAME="$1"
if [ -z "$CASE_NAME" ]; then
  echo "❌ Error: Please provide a case name, e.g."
  echo "   sbatch Combiprecip_lisflood.sh Zell_2m"
  exit 1
fi

BUILD_DIR="/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/${CASE_NAME}"
PAR_FILE="${BUILD_DIR}/${CASE_NAME}.par"
EXEC="/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/lisflood"

echo "Running case: $CASE_NAME"
echo "Parameter file: $PAR_FILE"
echo "Executable: $EXEC"
echo

# -----------------------------
# 1) Clean environment
# -----------------------------
echo "Purging modules..."
module purge

# -----------------------------
# 2) Load modules
# -----------------------------
echo "Loading modules..."
module load foss || exit 1
module load CMake || exit 1
module load netCDF/4.9.2-gompi-2024a || exit 1
module load CUDA || exit 1

# -----------------------------
# 3) Verify GPU access
# -----------------------------
echo
echo "GPU & CUDA info:"
which nvcc || true
nvcc --version | head -n3 || true
echo
nvidia-smi || true
echo

# -----------------------------
# 4) Activate environment
# -----------------------------
echo "Activating conda env: env_py311"
source /storage/homefs/ge24z347/mambaforge/etc/profile.d/conda.sh
conda activate env_py311 || exit 1
echo

# -----------------------------
# 4.5) Force single-thread CPU execution (CRITICAL for acc_nugrid stability)
# -----------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Make CUDA errors synchronous (helps reveal GPU illegal access instead of silent segfaults)
export CUDA_LAUNCH_BLOCKING=1

echo "Thread settings:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "  NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo

# -----------------------------
# 5) Background GPU monitoring
# -----------------------------
GPU_LOG="/storage/homefs/ge24z347/gpu_usage_${SLURM_JOB_ID}.log"
echo "GPU usage will be logged to $GPU_LOG"

monitor_gpu_usage() {
    while true; do
        nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total \
        --format=csv,noheader >> "$GPU_LOG"
        sleep 60
    done
}

sleep 2
monitor_gpu_usage &
MONITOR_PID=$!
trap "echo 'Stopping GPU monitor'; kill $MONITOR_PID 2>/dev/null" EXIT

# -----------------------------
# 6) Run the simulation
# -----------------------------
echo
echo "===== Starting LISFLOOD simulation at $(date) ====="
cd "$BUILD_DIR" || { echo "❌ ERROR: Directory not found: $BUILD_DIR"; exit 1; }

# Use verbose time to capture peak memory in the log
/usr/bin/time -v "$EXEC" "$PAR_FILE"
SIM_EXIT=$?

echo
echo "===== LISFLOOD finished with exit code: $SIM_EXIT ====="
echo

# -----------------------------
# 7) Usage reminder
# -----------------------------
echo "To get resource usage later:"
echo "  sacct --job $SLURM_JOB_ID --format=JobID,JobName,ReqMem,MaxRSS,Elapsed,State,ExitCode"
echo

echo "========== SLURM job finished at $(date) =========="
exit $SIM_EXIT
