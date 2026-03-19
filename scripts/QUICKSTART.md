# QUICK START GUIDE
# Road Sign Detector - Verification & Analysis Suite

## What You Have

A complete software-based verification suite with:

1. **Golden Model** - Reference implementation of all processing stages
2. **Comparison Tool** - Compares RTL to golden model with visualizations  
3. **Report Generator** - Detailed analysis and metrics
4. **Orchestrator** - Master script to run everything

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
python3 -m pip install numpy scipy matplotlib Pillow
```

### Step 2: Run Verification
```bash
cd /users/epnyrk/Project/design/work/ProjectA/scripts
python3 run_verification.py
```

### Step 3: Review Results
Open these files in the `results/` directory:
- `01_mask_comparison.png` - Red mask analysis
- `02_morph_comparison.png` - Morphology filter analysis
- `03_pipeline_flow.png` - Complete pipeline visualization
- `detailed_report.txt` - Detailed statistics

## What Each Script Does

| Script | Purpose | Output |
|--------|---------|--------|
| `golden_model_complete.py` | Generates reference outputs | `mask_out2.txt`, `morph_out2.txt`, etc. |
| `comparison_and_analysis.py` | Compares RTL to golden model | `01_mask_comparison.png`, `02_morph_comparison.png` |
| `detailed_report_generator.py` | Creates analysis reports | `detailed_report.txt`, `verification_report.json` |
| `run_verification.py` | Runs all scripts in sequence | All of the above |
| `setup_and_doctor.py` | Checks environment & dependencies | Diagnostic report |

## Key Features

✓ **Golden Model Implementation**
  - Stage 1: Red Mask Detection
  - Stage 2: Morphology Filter (Dilation + Erosion)
  - Stage 3: Connected Component Labeling
  - Stage 4: Geometry Filtering

✓ **Comprehensive Comparison**
  - Pixel-by-pixel analysis
  - False positive/negative detection
  - Statistical accuracy metrics

✓ **Detailed Visualizations**
  - Side-by-side comparison plots
  - Error heatmaps
  - Statistical charts and tables

✓ **Complete Reporting**
  - Text-based detailed report
  - JSON report for programmatic access
  - Per-stage metrics and analysis

## Metrics Provided

- **Accuracy** - % of correctly classified pixels
- **Precision** - True positives vs false positives
- **Recall** - Detection rate (false negatives)
- **F1-Score** - Balanced metric combining precision & recall
- **Confusion Matrix** - TP, TN, FP, FN counts
- **Error Regions** - Spatial analysis of mismatches

## File Locations

| Directory | Purpose |
|-----------|---------|
| `scripts/` | All verification scripts |
| `data/` | Input images and reference outputs |
| `results/` | Analysis plots and reports |

## Important Notes

⚠️ **Python Version**: Requires Python 3.7 or later
  - Check: `python3 --version`
  - If < 3.7, use: `python3.7` or `python3.8` instead of `python3`

⚠️ **RTL Outputs**: Comparison requires RTL simulation outputs
  - Golden model generation works independently
  - RTL outputs: `actual_mask_out.txt`, `actual_morph_out.txt`

⚠️ **Processing Time**: ~20-30 seconds for complete verification

## Troubleshooting

### Python version error
```bash
python3.7 run_verification.py
```

### Missing packages
```bash
python3 -m pip install numpy scipy matplotlib Pillow
```

### Check environment
```bash
python3 setup_and_doctor.py
```

## Example Output

After running `python3 run_verification.py`, you'll see:

```
✓ Step 1: Generate Golden Model Reference - PASS (5.2s)
✓ Step 2: Compare RTL Simulation vs Golden Model - PASS (3.8s)
✓ Step 3: Generate Detailed Analysis Reports - PASS (1.5s)

Generated Visualizations:
  - 01_mask_comparison.png
  - 02_morph_comparison.png
  - 03_pipeline_flow.png

Generated Reports:
  - detailed_report.txt
  - verification_report.json
```

## Next Steps

1. Run diagnostics: `python3 setup_and_doctor.py`
2. Generate golden model: `python3 golden_model_complete.py`
3. Compare RTL: `python3 comparison_and_analysis.py`
4. View results in `results/` directory

Or run everything at once:
```bash
python3 run_verification.py
```

For full documentation:
```bash
cat README.md
```

---

**Version**: 1.0  
**Created**: 2026-03-19  
**Location**: `/users/epnyrk/Project/design/work/ProjectA/scripts/`
