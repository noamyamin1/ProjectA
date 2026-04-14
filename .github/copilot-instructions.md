# Project Guidelines

## Code Style
- SystemVerilog RTL: explicit port connections, active-low async resets.
- Comments, docstrings, and commit messages in English only.

## Architecture
- Top-level RTL: road_sign_detector.sv with AXI-Lite, AXI-Full, AXI-Stream interfaces.
- Processing pipeline: red mask -> morphology -> CCL -> geometry filter -> template matching -> results writeback.
- Primary system TB: tb/road_sign_detector_top_tb.sv.

## Build and Test
- End-to-end flow: ./scripts/run_e2e_tb.sh [options] [image1 image2 ...]
  - Runs VCS from ~/Project, simv from repo root, writes results/simulation_output.log.
- Template-matching unit: ./scripts/run_roi_template.sh <path_to_image>

## Conventions
- Do not modify existing RTL files under rtl/ without explicit user permission.
- Python verification: activate venv before running scripts:
  - cd ~/Project/design/work/ProjectA && source ./.venv/bin/activate.csh
- Use images from pyton/pics_bank/ for testing/validation inputs.
- Template matching must follow pyton/low_level_version.ipynb (SAD + normalization).
- Keep scripts/ for automation only; avoid extra ad-hoc files.
