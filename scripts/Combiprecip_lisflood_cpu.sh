#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=cpu-invest
#SBATCH --qos=job_cpu_preemptable

#SBATCH --job-name="LISFLOOD_CPU_preempt"
#SBATCH --time=6:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

#SBATCH --output=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/scripts/logs/%x_%j.out
#SBATCH --error=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/scripts/logs/%x_%j.err

echo "========== SLURM job started at $(date) =========="

# -----------------------------
# 0) Read case name
# -----------------------------
CASE_NAME="$1"
if [ -z "$CASE_NAME" ]; then
  echo "❌ Error: Please provide a case name, e.g."
  echo "   sbatch Combiprecip_lisflood_cpu.sh Zell_2m"
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
# 1) Modules needed to RUN on CPU
# -----------------------------
module purge
module load foss || exit 1
module load netCDF/4.9.2-gompi-2024a || exit 1

# If you truly don't need Python for this run, you can remove conda entirely.
# Keep it only if your workflow requires it.
echo "Activating conda env: env_py311"
source /storage/homefs/ge24z347/mambaforge/etc/profile.d/conda.sh
conda activate env_py311 || exit 1
echo

# -----------------------------
# 2) Thread settings (CPU)
# -----------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Thread settings:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "  NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo

# -----------------------------
# 3) Run the simulation
# -----------------------------
echo "===== Starting LISFLOOD simulation at $(date) ====="
cd "$BUILD_DIR" || { echo "❌ ERROR: Directory not found: $BUILD_DIR"; exit 1; }

# Use verbose time to capture peak memory in the log
/usr/bin/time -v "$EXEC" "$PAR_FILE"
SIM_EXIT=$?

echo
echo "===== LISFLOOD finished with exit code: $SIM_EXIT ====="
echo

echo "To get resource usage later:"
echo "  sacct --job $SLURM_JOB_ID --format=JobID,JobName,ReqMem,MaxRSS,Elapsed,State,ExitCode"
echo

echo "========== SLURM job finished at $(date) =========="
exit $SIM_EXIT

