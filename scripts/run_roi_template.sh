#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./scripts/run_roi_template.sh <path_to_image>"
    echo "Example: ./scripts/run_roi_template.sh pyton/pics_bank/22.png"
    exit 1
fi

IMAGE_PATH=$1
PROJECT_A_DIR="/users/epnyrk/Project/design/work/ProjectA"
SIM_DIR="/users/epnyrk/Project"
TEMPLATE_DIR_DEFAULT="${PROJECT_A_DIR}/pyton/Templates_white"
if [ -d "${TEMPLATE_DIR_DEFAULT}" ]; then
    TEMPLATE_DIR="${TEMPLATE_DIR_DEFAULT}"
else
    TEMPLATE_DIR="${PROJECT_A_DIR}/pyton/Templates"
fi
SIM_LOG="${PROJECT_A_DIR}/results/simulation_output.log"

set -e

echo "========================================"
echo " STEP 1: Prepare Templates ROM"
echo "========================================"
python3 scripts/prep_templates_mem.py "${TEMPLATE_DIR}" data/templates.mem data/template_mapping.txt

echo ""
echo "========================================"
echo " STEP 2: Prepare ROI + Golden"
echo "========================================"
python3 scripts/prep_roi_template.py "${IMAGE_PATH}" data/

echo ""
echo "========================================"
echo " STEP 3: Compile RTL & TB (VCS)"
echo "========================================"
cd "${SIM_DIR}"
vcs -kdb -sverilog -debug_access+all -full64 -timescale=1ns/1ps \
    design/work/ProjectA/rtl/roi_fetcher_axi_master.sv \
    design/work/ProjectA/rtl/template_matching_engine.sv \
    design/work/ProjectA/tb/tb_roi_template_unit.sv

echo ""
echo "========================================"
echo " STEP 4: Run Simulation (SIMV)"
echo "========================================"
./simv |& tee "${SIM_LOG}"

echo ""
echo "========================================"
echo " STEP 5: Verify Results"
echo "========================================"
cd "${PROJECT_A_DIR}"
python3 scripts/verify_roi_template.py \
    data/template_matching_golden.txt \
    data/actual_template_matching.txt \
    results/ \
    "${IMAGE_PATH}" \
    "${TEMPLATE_DIR}"

echo ""
echo "Flow complete. Check results directory for reports and visuals."