#!/bin/bash

PROJECT_A_DIR="/users/epnyrk/Project/design/work/ProjectA"
SIM_DIR="/users/epnyrk/Project"

echo "========================================"
echo " STEP 1: Pre-processing (CCL Python Model)"
echo "========================================"
python3 scripts/prep_ccl.py data/actual_morph_out.txt data/
if [ $? -ne 0 ]; then echo "Pre-processing failed!"; exit 1; fi

echo ""
echo "========================================"
echo " STEP 2: Compiling RTL & TB (VCS)"
echo "========================================"
cd ${SIM_DIR}
vcs -kdb -sverilog -debug_access+all -full64 -ignore initializer_driver_checks \
    design/work/ProjectA/rtl/ccl_pass1_labeler.sv \
    design/work/ProjectA/rtl/ccl_uf_resolver.sv \
    design/work/ProjectA/rtl/ccl_engine.sv \
    design/work/ProjectA/tb/tb_ccl_engine_debug.sv
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
python3 scripts/verify_ccl.py \
    data/ccl_pass1_golden.txt \
    data/actual_ccl_pass1.txt \
    data/ccl_pass2_golden.txt \
    data/actual_ccl_pass2.txt \
    results/

echo ""
echo "Flow complete! Check the terminal output for the match results."