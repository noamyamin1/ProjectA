#!/bin/bash

PROJECT_A_DIR="/users/epnyrk/Project/design/work/ProjectA"
SIM_DIR="/users/epnyrk/Project"
TEMPLATE_DIR_DEFAULT="${PROJECT_A_DIR}/pyton/Templates_white"
if [ -d "${TEMPLATE_DIR_DEFAULT}" ]; then
    TEMPLATE_DIR="${TEMPLATE_DIR_DEFAULT}"
else
    TEMPLATE_DIR="${PROJECT_A_DIR}/pyton/Templates"
fi
SUMMARY_PATH="${PROJECT_A_DIR}/results/roi_template_batch_summary.txt"
EXPECTED_PATH="${PROJECT_A_DIR}/data/expected_results.txt"

set -e

IMAGES=("$@")
if [ ${#IMAGES[@]} -eq 0 ]; then
    IMAGES=("${PROJECT_A_DIR}/pyton/pics_bank"/*.{png,jpg,jpeg})
fi

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
SKIP_STEMS_STATIC=(
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
    for item in "${SKIP_STEMS_STATIC[@]}"; do
        if [ "${stem}" = "${item}" ]; then
            return 0
        fi
    done
    return 1
}

mkdir -p "${PROJECT_A_DIR}/results"
: > "${SUMMARY_PATH}"

SKIP_STEMS=()
if [ -f "${EXPECTED_PATH}" ]; then
    while IFS= read -r line; do
        case "${line}" in
            *"do not try validating"*)
                key="${line%%-*}"
                key="$(echo "${key}" | xargs)"
                if [ -n "${key}" ]; then
                    SKIP_STEMS+=("${key}")
                    if [ "${key}" = "No_entrance" ]; then
                        SKIP_STEMS+=("No_enterance")
                    fi
                fi
                ;;
        esac
    done < "${EXPECTED_PATH}"
fi

echo "ROI+Template batch run" | tee -a "${SUMMARY_PATH}"

echo "========================================"
echo " STEP 0: Prepare Templates ROM (once)"
echo "========================================"
cd "${PROJECT_A_DIR}"
python3 scripts/prep_templates_mem.py "${TEMPLATE_DIR}" data/templates.mem data/template_mapping.txt

for IMG in "${IMAGES[@]}"; do
    if [ ! -f "${IMG}" ]; then
        continue
    fi
    if is_skipped_image "${IMG}"; then
        echo "Skipping excluded image: ${IMG}"
        continue
    fi
    echo "========================================"
    echo " RUN ROI+TM FOR: ${IMG}"
    echo "========================================"

    STEM=$(basename "${IMG}")
    STEM="${STEM%.*}"
    SKIP_THIS=0
    for SKIP in "${SKIP_STEMS[@]}"; do
        if [ "${STEM}" = "${SKIP}" ]; then
            SKIP_THIS=1
            break
        fi
    done
    if [ ${SKIP_THIS} -eq 1 ]; then
        echo "${IMG} roi_template: SKIP (do not validate)" | tee -a "${SUMMARY_PATH}"
        continue
    fi
    ARCHIVE_DIR="${PROJECT_A_DIR}/results/by_image/${STEM}"
    mkdir -p "${ARCHIVE_DIR}"

    echo "[1/2] Ensure per-image pipeline outputs (morph/CCL)"
    set +e
    ./scripts/run_stats_geom.sh "${IMG}"
    STATS_RC=$?
    set -e

    if [ ${STATS_RC} -ne 0 ]; then
        echo "${IMG} stats_geom: FAIL" | tee -a "${SUMMARY_PATH}"
        continue
    fi

    ROI_COUNT=$(grep -v "^[[:space:]]*#" "${PROJECT_A_DIR}/data/geom_bboxes_golden.txt" 2>/dev/null | wc -l)
    if [ ${ROI_COUNT} -eq 0 ]; then
        echo "${IMG} roi_template: SKIP (no ROIs after geom filter)" | tee -a "${SUMMARY_PATH}"
        printf "# roi_id xmin xmax ymin ymax label\n" > "${PROJECT_A_DIR}/data/roi_list.txt"
        printf "# roi_id xmin xmax ymin ymax best_class_id best_score\n" > "${PROJECT_A_DIR}/data/template_matching_golden.txt"
        : > "${PROJECT_A_DIR}/data/actual_template_matching.txt"
        cp -f "${PROJECT_A_DIR}/data/roi_list.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/template_matching_golden.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/actual_template_matching.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/geom_bboxes_golden.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/geom_filtered_golden.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/geom_filter_debug.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        cp -f "${PROJECT_A_DIR}/data/actual_geom_boxes.txt" "${ARCHIVE_DIR}/" 2>/dev/null
        continue
    fi

    echo "[2/2] ROI fetch + template match"
    set +e
    ./scripts/run_roi_template.sh "${IMG}"
    TM_RC=$?
    set -e

    # Archive outputs for traceability
    set +e
    cp -f "${PROJECT_A_DIR}/data/template_matching_golden.txt" "${ARCHIVE_DIR}/" 2>/dev/null
    cp -f "${PROJECT_A_DIR}/data/actual_template_matching.txt" "${ARCHIVE_DIR}/" 2>/dev/null
    cp -f "${PROJECT_A_DIR}/data/roi_list.txt" "${ARCHIVE_DIR}/" 2>/dev/null
    cp -f "${PROJECT_A_DIR}/data/actual_top3.txt" "${ARCHIVE_DIR}/" 2>/dev/null
    cp -f "${PROJECT_A_DIR}/data/actual_scores_full.txt" "${ARCHIVE_DIR}/" 2>/dev/null
    cp -f "${PROJECT_A_DIR}/data/actual_roi_bin_0.txt" "${ARCHIVE_DIR}/" 2>/dev/null
    cp -f "${PROJECT_A_DIR}/data/golden_roi_bin_0.txt" "${ARCHIVE_DIR}/" 2>/dev/null
    cp -f "${PROJECT_A_DIR}/results/template_matching_verify.txt" "${ARCHIVE_DIR}/" 2>/dev/null
    cp -f "${PROJECT_A_DIR}/results/template_matching_debug_0.png" "${ARCHIVE_DIR}/" 2>/dev/null
    cp -f "${PROJECT_A_DIR}/results/simulation_output.log" "${ARCHIVE_DIR}/" 2>/dev/null

    # Re-generate debug PNG using archived inputs to avoid cross-image contamination
    if [ -f "${ARCHIVE_DIR}/template_matching_golden.txt" ] && [ -f "${ARCHIVE_DIR}/actual_template_matching.txt" ]; then
        python3 scripts/verify_roi_template.py \
            "${ARCHIVE_DIR}/template_matching_golden.txt" \
            "${ARCHIVE_DIR}/actual_template_matching.txt" \
            "${ARCHIVE_DIR}" \
            "${IMG}" \
            "${TEMPLATE_DIR}" --no-exit
    fi
    set -e

    if [ ${TM_RC} -ne 0 ]; then
        echo "${IMG} roi_template: FAIL" | tee -a "${SUMMARY_PATH}"
    else
        echo "${IMG} roi_template: PASS" | tee -a "${SUMMARY_PATH}"
    fi

done

echo "========================================"
echo " STEP 3: Compare Expected vs Actual"
echo "========================================"
cd "${PROJECT_A_DIR}"
python3 scripts/report_expected_results.py

echo "Batch complete. Summary: ${SUMMARY_PATH}"
