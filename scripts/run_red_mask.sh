#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./scripts/run_red_mask.sh <path_to_image>"
    echo "Example: ./scripts/run_red_mask.sh pyton/pics_bank/test1.jpg"
    exit 1
fi

IMAGE_PATH=$1
PROJECT_A_DIR="/users/epnyrk/Project/design/work/ProjectA"
SIM_DIR="/users/epnyrk/Project"

echo "========================================"
echo " STEP 1: Pre-processing (Python)"
echo "========================================"
python3 scripts/prep_red_mask.py ${IMAGE_PATH} data/
if [ $? -ne 0 ]; then echo "Pre-processing failed!"; exit 1; fi

echo ""
echo "========================================"
echo " STEP 2: Compiling RTL & TB (VCS)"
echo "========================================"
cd ${SIM_DIR}
vcs -kdb -sverilog -debug_access+all -full64 \
    design/work/ProjectA/rtl/red_mask_datapath.sv \
    design/work/ProjectA/tb/tb_red_mask_datapath_image.sv
if [ $? -ne 0 ]; then echo "Compilation failed!"; exit 1; fi

echo ""
echo "========================================"
echo " STEP 3: Running Simulation (SIMV)"
echo "========================================"
./simv
if [ $? -ne 0 ]; then echo "Simulation failed!"; exit 1; fi

echo ""
echo "========================================"
echo " STEP 4: Post-processing & Verification"
echo "========================================"
cd ${PROJECT_A_DIR}
python3 scripts/verify_red_mask.py \
    data/mask_out.txt \
    data/actual_mask_out.txt \
    results/

echo ""
echo "Flow complete! Check the results directory for visualization."