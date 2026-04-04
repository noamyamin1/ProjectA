#!/bin/bash

PROJECT_A_DIR="/users/epnyrk/Project/design/work/ProjectA"
SIM_DIR="/users/epnyrk/Project"

echo "========================================"
echo " STEP 1: Pre-processing (Morphology)"
echo "========================================"
python3 scripts/prep_morphology.py data/mask_out.txt data/
if [ $? -ne 0 ]; then echo "Pre-processing failed!"; exit 1; fi

echo ""
echo "========================================"
echo " STEP 2: Compiling RTL & TB (VCS)"
echo "========================================"
cd ${SIM_DIR}
vcs -kdb -sverilog -debug_access+all -full64 \
    design/work/ProjectA/rtl/sliding_window_3x3.sv \
    design/work/ProjectA/rtl/morphology_filter.sv \
    design/work/ProjectA/tb/tb_morphology_filter_image.sv
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
python3 scripts/verify_morphology.py \
    data/morph_out.txt \
    data/actual_morph_out.txt \
    results/

echo ""
echo "Flow complete! Check the results directory for morphology visualization."