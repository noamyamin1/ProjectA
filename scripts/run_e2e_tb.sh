#!/bin/bash

set -e

PROJECT_A_DIR="/users/epnyrk/Project/design/work/ProjectA"
SIM_DIR="/users/epnyrk/Project"
TEMPLATE_DIR_DEFAULT="${PROJECT_A_DIR}/pyton/Templates_white"
SIM_LOG="${PROJECT_A_DIR}/results/simulation_output.log"

if [ -d "${TEMPLATE_DIR_DEFAULT}" ]; then
    TEMPLATE_DIR="${TEMPLATE_DIR_DEFAULT}"
else
    TEMPLATE_DIR="${PROJECT_A_DIR}/pyton/Templates"
fi

PY="python3"
if [ -x "${PROJECT_A_DIR}/.venv/bin/python" ]; then
    PY="${PROJECT_A_DIR}/.venv/bin/python"
fi

usage() {
    echo "Usage: ./scripts/run_e2e_tb.sh [--continuous] [--fast] [--no-compile] [--no-prep] [--no-visuals] [image1 image2 ...]"
    echo "If no images are given, runs all images in pyton/pics_bank."
    echo "  --continuous: run multi-frame without +check_clear (continuous streaming)"
    echo "  --fast: skip compile, goldens prep, and visuals if possible"
    echo "  --no-compile: skip VCS compile if simv already exists"
    echo "  --no-prep: skip goldens prep (keeps existing files)"
    echo "  --no-visuals: skip visualization step"
}

CONTINUOUS=0
SKIP_COMPILE=0
SKIP_PREP=0
SKIP_VISUALS=0
IMAGES=()

SKIP_FILES=(
    "WhatsApp Image 2026-01-05 at 10.39.09.jpeg"
    "NO_entrance.jpg"
    "No_entrance.jpg"
    "No_enterance.jpg"
    "slippery_road.jpg"
    "down_triangle.jpg"
    "bumpers.jpg"
    "10.png"
    "1.jpeg"
    "3.jpeg"
    "4.jpeg"
    "9.png"
    "8.png"
    "26.png"
)
SKIP_STEMS=(
    "WhatsApp Image 2026-01-05 at 10.39.09"
    "NO_entrance"
    "No_entrance"
    "No_enterance"
    "bumpers"
    "10"
    "8"
    "slippery_road"
    "down_triangle"
    "1"
    "3"
    "4"
    "9"
    "26"
)

is_skipped_image() {
    local path="$1"
    local base
    local stem
    base=$(basename "${path}")
    stem="${base%.*}"
    for item in "${SKIP_FILES[@]}"; do
        if [ "${base}" = "${item}" ]; then
            return 0
        fi
    done
    for item in "${SKIP_STEMS[@]}"; do
        if [ "${stem}" = "${item}" ]; then
            return 0
        fi
    done
    return 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --continuous|--no-check-clear)
            CONTINUOUS=1
            shift
            ;;
        --fast)
            SKIP_COMPILE=1
            SKIP_PREP=1
            SKIP_VISUALS=1
            shift
            ;;
        --no-compile)
            SKIP_COMPILE=1
            shift
            ;;
        --no-prep)
            SKIP_PREP=1
            shift
            ;;
        --no-visuals)
            SKIP_VISUALS=1
            shift
            ;;
        *)
            IMAGES+=("$1")
            shift
            ;;
    esac
done

shopt -s nullglob
if [ ${#IMAGES[@]} -eq 0 ]; then
    IMAGES=("${PROJECT_A_DIR}/pyton/pics_bank"/*.png "${PROJECT_A_DIR}/pyton/pics_bank"/*.jpg "${PROJECT_A_DIR}/pyton/pics_bank"/*.jpeg)
fi
shopt -u nullglob

FILTERED_IMAGES=()
for IMG in "${IMAGES[@]}"; do
    if is_skipped_image "${IMG}"; then
        echo "Skipping excluded image: ${IMG}"
        continue
    fi
    FILTERED_IMAGES+=("${IMG}")
done
IMAGES=("${FILTERED_IMAGES[@]}")

if [ ${#IMAGES[@]} -eq 0 ]; then
    echo "No images found (after exclusions)."
    exit 1
fi

mkdir -p "${PROJECT_A_DIR}/results"

FRAME_NAMES=()
FRAMES_JOINED=""

if [ ${SKIP_PREP} -eq 0 ] || [ ! -f "${PROJECT_A_DIR}/data/templates.mem" ]; then
    echo "========================================"
    echo "STEP 0: Prepare Templates ROM"
    echo "========================================"
    cd "${PROJECT_A_DIR}"
    ${PY} scripts/prep_templates_mem.py "${TEMPLATE_DIR}" data/templates.mem data/template_mapping.txt
fi

for IMG in "${IMAGES[@]}"; do
    if [ ! -f "${IMG}" ]; then
        echo "Skipping missing image: ${IMG}"
        continue
    fi

    STEM=$(basename "${IMG}")
    FRAME_NAMES+=("${STEM}")
    STEM="${STEM%.*}"
    ARCHIVE_DIR="${PROJECT_A_DIR}/results/by_image/${STEM}"
    mkdir -p "${ARCHIVE_DIR}"

    if [ ${SKIP_PREP} -eq 0 ]; then
        echo "========================================"
        echo "PREP GOLDENS FOR: ${IMG}"
        echo "========================================"

        ./scripts/run_stats_geom.sh "${IMG}"

        MORPH_SRC="${ARCHIVE_DIR}/morph_out.txt"
        if [ ! -f "${MORPH_SRC}" ]; then
            MORPH_SRC="${PROJECT_A_DIR}/data/morph_out.txt"
        fi
        if [ -f "${MORPH_SRC}" ]; then
            ${PY} scripts/prep_ccl.py "${MORPH_SRC}" "${ARCHIVE_DIR}"
        fi

        set +e
        ${PY} scripts/prep_roi_template.py "${IMG}" "${ARCHIVE_DIR}"
        PREP_TM_RC=$?
        set -e
        if [ ${PREP_TM_RC} -ne 0 ]; then
            echo "Warning: prep_roi_template failed for ${IMG} (no ROI or missing bbox)."
        fi
    fi

    ${PY} scripts/gen_image_hex.py "${IMG}"

done

MODE_ARG=""
CHECK_CLEAR_ARG=""
FRAME_LIST_FILE=""

FRAMES_JOINED=$(IFS=,; echo "${FRAME_NAMES[*]}")

cleanup_frame_list() {
    if [ -n "${FRAME_LIST_FILE}" ] && [ -f "${FRAME_LIST_FILE}" ]; then
        rm -f "${FRAME_LIST_FILE}"
    fi
}

if [ ${#FRAME_NAMES[@]} -eq 1 ]; then
    MODE_ARG="+mode=single +single=${FRAME_NAMES[0]}"
else
    FRAME_LIST_FILE=$(mktemp /tmp/projecta_frames_XXXXXX.txt)
    for name in "${FRAME_NAMES[@]}"; do
        printf "%s\n" "${name}" >> "${FRAME_LIST_FILE}"
    done
    MODE_ARG="+mode=multi +frame_list_file=${FRAME_LIST_FILE}"
    if [ ${CONTINUOUS} -eq 1 ]; then
        CHECK_CLEAR_ARG="+check_clear=0"
    else
        CHECK_CLEAR_ARG="+check_clear=1"
    fi
    trap cleanup_frame_list EXIT
fi

if [ ${SKIP_COMPILE} -eq 0 ] || [ ! -x "${SIM_DIR}/simv" ]; then
    echo "========================================"
    echo "STEP 1: Compile E2E TB"
    echo "========================================"
    cd "${SIM_DIR}"
    vcs -kdb -sverilog -debug_access+all -full64 -timescale=1ns/1ps \
        design/work/ProjectA/rtl/*.sv \
        design/work/ProjectA/tb/road_sign_detector_top_tb.sv
else
    echo "Skipping compile (simv exists)"
fi

echo "========================================"
echo "STEP 2: Run E2E TB"
echo "========================================"
cd "${PROJECT_A_DIR}"
${SIM_DIR}/simv ${MODE_ARG} ${CHECK_CLEAR_ARG} |& tee "${SIM_LOG}"

if [ ${SKIP_VISUALS} -eq 0 ]; then
    echo "========================================"
    echo "STEP 3: Generate E2E Visuals"
    echo "========================================"
    cd "${PROJECT_A_DIR}"
    if [ -n "${FRAME_LIST_FILE}" ] && [ -f "${FRAME_LIST_FILE}" ]; then
        ${PY} scripts/verify_e2e_visuals.py --frame-list-file="${FRAME_LIST_FILE}"
    else
        ${PY} scripts/verify_e2e_visuals.py --frames="${FRAMES_JOINED}"
    fi
else
    echo "Skipping visuals"
fi

echo "========================================"
echo "STEP 4: Summarize E2E Results"
echo "========================================"
cd "${PROJECT_A_DIR}"
${PY} scripts/summarize_e2e_results.py --out "${PROJECT_A_DIR}/results/e2e_frame_summary.txt"

echo "========================================"
echo "DONE"
echo "Summary: results/e2e_summary.txt"
echo "Per-frame visuals: results/by_image/<stem>/"
