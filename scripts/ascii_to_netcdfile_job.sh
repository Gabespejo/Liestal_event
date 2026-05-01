#!/bin/bash
#SBATCH --account=gratis
#SBATCH --job-name=ascii2nc_bellinzona
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=ascii2nc_%j.out
#SBATCH --error=ascii2nc_%j.err

source /storage/homefs/ge24z347/mambaforge/etc/profile.d/conda.sh
conda activate env_py311

cd /storage/homefs/ge24z347/Zell_event/scripts

python ./ascii_to_netcdfile.py \
  --dir "/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Bellinzona_2m_accv/Bellinzona_2m_accv" \
  --base Bellinzona_2m_accv \
  --start 0 --end 10 \
  --reference-start "2021-08-07T13:00:00.000000000" \
  --dt-minutes 60 \
  --compute-derived \
  --out "/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/Bellinzona_2m_accv/Bellinzona_2m_accv_Combiprecip.nc" \
  --force