#!/bin/bash
#------------------------
#SBATCH --account=gratis
#------------------------
#SBATCH --partition=cpu-invest
#SBATCH --qos=job_cpu_preemptable

#SBATCH --job-name="Preemptable Job example"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Zell_50cm_accnugrid/scripts/logs/%x_%j.out
#SBATCH --error=/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Zell_50cm_accnugrid/scripts/%x_%j.err

module load ParaView/5.11.2-foss-2023a

# If you need your conda python for rasterio, activate it robustly:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_py311

# ---- HARD disable UCX/Infiniband usage ----
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp,vader
export OMPI_MCA_btl_tcp_if_include=lo
export UCX_TLS=self,sm,tcp
export UCX_NET_DEVICES=lo
export UCX_IB_DISABLE=1

# Optional: reduce noise
export UCX_LOG_LEVEL=warn

# ---- sanity test: can pvpython start at all? ----
mpirun -n 1 pvpython -c "print('pvpython_ok')"

# ---- run your converter script ----
python /storage/homefs/ge24z347/Zell_event/scripts/vtkfiles_to_tif.py \
  /storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Zell_50cm_accnugrid/Zell_50cm_accnugrid/Zell_50cm-0008.vtk\
  h \
  /storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Zell_50cm_accnugrid/Zell_50cm-0008_lv95.tif \
  0.5 2056

