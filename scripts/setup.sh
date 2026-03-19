#!/bin/bash
# Quick installation and testing script for the verification suite

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║   Road Sign Detector - Verification Suite Quick Setup              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Define paths
PROJECT_DIR="/users/epnyrk/Project/design/work/ProjectA"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
DATA_DIR="${PROJECT_DIR}/data"
RESULTS_DIR="${PROJECT_DIR}/results"

# Change to scripts directory
cd "${SCRIPTS_DIR}" || exit 1

echo "[1] Running diagnostic checks..."
python3 setup_and_doctor.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Diagnostic checks failed"
    echo "Please fix the issues above before proceeding"
    exit 1
fi

echo ""
echo "[2] Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo "  Option A: Run full verification (recommended)"
echo "    $ cd ${SCRIPTS_DIR}"
echo "    $ python3 run_verification.py"
echo ""
echo "  Option B: Generate golden model only"
echo "    $ python3 golden_model_complete.py"
echo ""
echo "  Option C: Run individual scripts"
echo "    $ python3 golden_model_complete.py"
echo "    $ python3 comparison_and_analysis.py"
echo "    $ python3 detailed_report_generator.py"
echo ""
echo "  Option D: View help/documentation"
echo "    $ cat README.md"
echo ""
echo "Data will be saved to:"
echo "  - Reference data: ${DATA_DIR}/"
echo "  - Analysis plots: ${RESULTS_DIR}/"
echo ""
