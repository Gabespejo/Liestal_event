#!/bin/bash

# Usage: ./run_dg2.sh Zell_2m_dg2

# 0) Get case name from input
CASE_NAME="$1"
if [ -z "$CASE_NAME" ]; then
  echo "❌ Error: You must provide a case name (e.g. ./run_dg2.sh Zell_2m_dg2)"
  exit 1
fi

# 1) Define directories
BUILD_DIR="/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/${CASE_NAME}"
PAR_FILE="${BUILD_DIR}/${CASE_NAME}.par"
EXEC_DEM="/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/generateDG2DEM"
EXEC_START="/storage/homefs/ge24z347/LISFLOOD_FP_8_1/build/generateDG2start"

# 2) Start clean
echo "🧹 Purging environment modules..."
module purge

# 3) Load required libraries
echo "📦 Loading modules..."
module load foss || exit 1
module load CMake || exit 1
module load netCDF/4.9.2-gompi-2023a || exit 1
module load CUDA || exit 1

# 4) Run generateDG2DEM
echo "⚙️ Running generateDG2DEM for ${CASE_NAME}..."
cd "$BUILD_DIR" || exit 1
$EXEC_DEM "$PAR_FILE"

echo "✔ Finished generateDG2DEM."

# 5) Run generateDG2start
echo "⚙️ Running generateDG2start for ${CASE_NAME}..."
$EXEC_START "$PAR_FILE"

echo "✔ Finished generateDG2start."
echo "✔ All DG2 preprocessing done. Files are in: $BUILD_DIR"