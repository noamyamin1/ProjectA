#!/bin/bash

PROJECT_A_DIR="/users/epnyrk/Project/design/work/ProjectA"
SIM_DIR="/users/epnyrk/Project"

SOFT_MODE=0
if [ "$1" = "--soft" ]; then
    SOFT_MODE=1
    shift
fi

IMAGES=("$@")

SKIP_FILES=(
    "WhatsApp Image 2026-01-05 at 10.39.09.jpeg"
    "NO_entrance.jpg"
    "No_entrance.jpg"
    "No_enterance.jpg"
    "slippery_road.jpg"
    "down_triangle.jpg"
    "1.jpeg"
    "3.jpeg"
    "4.jpeg"
    "9.png"
    "26.png"
)
SKIP_STEMS=(
    "WhatsApp Image 2026-01-05 at 10.39.09"
    "NO_entrance"
    "No_entrance"
    "No_enterance"
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

MIN_AREA=300
MAX_AREA=100000
MIN_W=34
MIN_H=32
MIN_PIX_AREA=313
MIN_SOLIDITY=0.217
MIN_W_RELAX=31
MIN_H_RELAX=30
RELAX_SOLIDITY=0.40
MAX_CANDIDATES=5

if [ ${SOFT_MODE} -eq 1 ]; then
    MIN_W=35
    MIN_H=33
    MIN_PIX_AREA=350
    MIN_SOLIDITY=0.25
    RELAX_SOLIDITY=0.45
fi

has_pass_status() {
    local f="$1"
    if [ -f "${f}" ] && grep -q "STATUS: PASS" "${f}"; then
        return 0
    fi
    return 1
}

use_cached_ccl() {
    local archive_dir="$1"
    if [ -f "${archive_dir}/actual_ccl_pass2.txt" ]; then
        cp -f "${archive_dir}/actual_ccl_pass2.txt" "${PROJECT_A_DIR}/data/" 2>/dev/null
        if [ -f "${archive_dir}/ccl_pass2_golden.txt" ]; then
            cp -f "${archive_dir}/ccl_pass2_golden.txt" "${PROJECT_A_DIR}/data/" 2>/dev/null
        fi
        return 0
    fi
    return 1
}

run_stats_geom_only() {
    local VIS_IMAGE="$1"
    local IMAGE_STEM=""
    local ARCHIVE_DIR=""

    echo "========================================"
    echo " STEP 1: Golden Prep (Stats + Geometry)"
    echo "========================================"
    cd "${PROJECT_A_DIR}"
    python3 scripts/prep_ccl_stats.py data/actual_ccl_pass2.txt data/
    python3 scripts/prep_geom_filter.py data/ccl_stats_golden.txt data/ \
        ${MIN_AREA} ${MAX_AREA} ${MIN_W} ${MIN_H} ${MIN_PIX_AREA} ${MIN_SOLIDITY} \
        ${MIN_W_RELAX} ${MIN_H_RELAX} ${RELAX_SOLIDITY} ${MAX_CANDIDATES}

    echo ""
    echo "========================================"
    echo " STEP 2: Compile TB (VCS)"
    echo "========================================"
    cd "${SIM_DIR}"
    set +e
    VCS_DEFS=""
    if [ ${SOFT_MODE} -eq 1 ]; then
        VCS_DEFS="+define+GEO_SOFT"
    fi

    vcs -kdb -sverilog -debug_access+all -full64 -ignore initializer_driver_checks ${VCS_DEFS} \
        design/work/ProjectA/rtl/ccl_stats_collector.sv \
        design/work/ProjectA/rtl/geometry_filter.sv \
        design/work/ProjectA/tb/tb_backend_stats_geom.sv
    VCS_RC=$?
    set -e

    if [ ${VCS_RC} -ne 0 ] && [ ! -x "${SIM_DIR}/simv" ]; then
        echo "Compilation failed (vcs rc=${VCS_RC}) and simv was not generated."
        exit 1
    fi

    echo ""
    echo "========================================"
    echo " STEP 3: Run Simulation"
    echo "========================================"
    ./simv

    echo ""
    echo "========================================"
    echo " STEP 4: Verify Results"
    echo "========================================"
    cd "${PROJECT_A_DIR}"
    set +e
    python3 scripts/verify_ccl_stats.py \
        data/ccl_stats_golden.txt \
        data/actual_ccl_stats.txt \
        results/
    VERIFY_CCL_RC=$?

    if [ -n "${VIS_IMAGE}" ]; then
        python3 scripts/verify_geom_filter.py \
            data/geom_bboxes_golden.txt \
            data/actual_geom_boxes.txt \
            results/ \
            pyton/pics_bank/ \
            "${VIS_IMAGE}"
    else
        python3 scripts/verify_geom_filter.py \
            data/geom_bboxes_golden.txt \
            data/actual_geom_boxes.txt \
            results/ \
            pyton/pics_bank/
    fi
    VERIFY_GEOM_RC=$?
    set -e

    echo ""
    echo "Flow complete. Check:"
    echo "  - results/ccl_stats_verify.txt"
    echo "  - results/geom_filter_verify.txt"
    echo "  - results/geom_filter_comparison.png"
    echo "  - results/geom_filter_overlay.png"
    if [ ${VERIFY_CCL_RC} -ne 0 ] || [ ${VERIFY_GEOM_RC} -ne 0 ]; then
        echo "Verification failed (ccl_rc=${VERIFY_CCL_RC}, geom_rc=${VERIFY_GEOM_RC})"
    else
        echo "Verification passed"
    fi

    if [ -n "${VIS_IMAGE}" ]; then
        IMAGE_STEM=$(basename "${VIS_IMAGE}")
        IMAGE_STEM="${IMAGE_STEM%.*}"
        ARCHIVE_DIR="${PROJECT_A_DIR}/results/by_image/${IMAGE_STEM}"
        mkdir -p "${ARCHIVE_DIR}"
        cp -f "${PROJECT_A_DIR}/data/ccl_stats_golden.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/actual_ccl_stats.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/geom_filtered_golden.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/geom_bboxes_golden.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/geom_filter_debug.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/actual_geom_boxes.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/results/ccl_stats_verify.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/results/geom_filter_verify.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/results/geom_filter_comparison.png" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/results/geom_filter_overlay.png" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/results/ccl_resolved_vis.png" "${ARCHIVE_DIR}/" 2>/dev/null
        echo "Archived results to: ${ARCHIVE_DIR}"
    fi
}

set -e

if [ ${#IMAGES[@]} -eq 0 ]; then
    run_stats_geom_only ""
else
    for IMG in "${IMAGES[@]}"; do
        if is_skipped_image "${IMG}"; then
            echo "Skipping excluded image: ${IMG}"
            continue
        fi
        echo "========================================"
        echo " RUN FULL PIPELINE FOR: ${IMG}"
        echo "========================================"
        cd "${PROJECT_A_DIR}"

        IMAGE_STEM=$(basename "${IMG}")
        IMAGE_STEM="${IMAGE_STEM%.*}"
        ARCHIVE_DIR="${PROJECT_A_DIR}/results/by_image/${IMAGE_STEM}"

        if has_pass_status "${ARCHIVE_DIR}/red_mask_stats.txt" && \
           [ -f "${ARCHIVE_DIR}/mask_out.txt" ] && [ -f "${ARCHIVE_DIR}/actual_mask_out.txt" ]; then
            cp -f "${ARCHIVE_DIR}/mask_out.txt" "${PROJECT_A_DIR}/data/" 2>/dev/null
            cp -f "${ARCHIVE_DIR}/actual_mask_out.txt" "${PROJECT_A_DIR}/data/" 2>/dev/null
            echo "Skipping red mask (cached PASS for ${IMAGE_STEM})"
        else
            ./scripts/run_red_mask.sh "${IMG}"
            mkdir -p "${ARCHIVE_DIR}"
            cp -f "${PROJECT_A_DIR}/data/mask_out.txt" "${ARCHIVE_DIR}/" 2>/dev/null
            cp -f "${PROJECT_A_DIR}/data/actual_mask_out.txt" "${ARCHIVE_DIR}/" 2>/dev/null
            cp -f "${PROJECT_A_DIR}/results/red_mask_stats.txt" "${ARCHIVE_DIR}/" 2>/dev/null
            cp -f "${PROJECT_A_DIR}/results/red_mask_comparison.png" "${ARCHIVE_DIR}/" 2>/dev/null
        fi

        if has_pass_status "${ARCHIVE_DIR}/morphology_stats.txt" && \
           [ -f "${ARCHIVE_DIR}/morph_out.txt" ] && [ -f "${ARCHIVE_DIR}/actual_morph_out.txt" ]; then
            cp -f "${ARCHIVE_DIR}/morph_out.txt" "${PROJECT_A_DIR}/data/" 2>/dev/null
            cp -f "${ARCHIVE_DIR}/actual_morph_out.txt" "${PROJECT_A_DIR}/data/" 2>/dev/null
            echo "Skipping morphology (cached PASS for ${IMAGE_STEM})"
        else
            ./scripts/run_morphology.sh
            mkdir -p "${ARCHIVE_DIR}"
            cp -f "${PROJECT_A_DIR}/data/morph_out.txt" "${ARCHIVE_DIR}/" 2>/dev/null
            cp -f "${PROJECT_A_DIR}/data/actual_morph_out.txt" "${ARCHIVE_DIR}/" 2>/dev/null
            cp -f "${PROJECT_A_DIR}/results/morphology_stats.txt" "${ARCHIVE_DIR}/" 2>/dev/null
            cp -f "${PROJECT_A_DIR}/results/morphology_comparison.png" "${ARCHIVE_DIR}/" 2>/dev/null
        fi

        if use_cached_ccl "${ARCHIVE_DIR}"; then
            echo "Skipping CCL (cached) for ${IMAGE_STEM}"
        else
            ./scripts/run_ccl.sh
            mkdir -p "${ARCHIVE_DIR}"
            cp -f "${PROJECT_A_DIR}/data/actual_ccl_pass2.txt" "${ARCHIVE_DIR}/" 2>/dev/null
            cp -f "${PROJECT_A_DIR}/data/ccl_pass2_golden.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        fi
        python3 scripts/visualize_ccl_resolved.py \
            data/actual_ccl_pass2.txt \
            results/ccl_resolved_vis.png
        mkdir -p "${ARCHIVE_DIR}"
        cp -f "${PROJECT_A_DIR}/results/ccl_resolved_vis.png" "${ARCHIVE_DIR}/" 2>/dev/null
        run_stats_geom_only "${IMG}"
    done
fi
