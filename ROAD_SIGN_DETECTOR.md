# Road Sign Detector: RTL + Verification Overview

## Overview
The road sign detector is a streaming RTL pipeline that ingests RGB frames over AXI-Stream, detects red sign candidates, extracts a region of interest, and classifies the sign using template matching. The top-level block exposes AXI-Lite control/status and AXI-Full memory interfaces for frame storage and ROI fetch. The verification environment uses both unit-level testbenches and a full end-to-end (E2E) testbench to compare intermediate streams against golden references and to validate the final classification.

## RTL Architecture and Submodules
### Top-Level
- **Top module**: [rtl/road_sign_detector.sv](rtl/road_sign_detector.sv)
- **Interfaces**: AXI-Lite (control/status), AXI-Stream (RGB input), AXI-Full masters (frame writeback and ROI fetch), `irq` output, and `current_id` output.
- **Frame flow**: The input stream is forked to the RGB frame writer and the red-mask pipeline. `frame_active` is controlled by start-of-frame (`tuser`) and cleared when the backend signals done.

### Submodules by Stage
1. **CSR and Control**
   - [rtl/csr_unit.sv](rtl/csr_unit.sv)
   - Provides configuration registers (`min_red_val`, `margin_shift`, `frame_base_addr`, `enable`) and exposes status (`best_class_id`, bbox, done flag). Generates `irq` on completion.

2. **RGB Frame Writer (AXI-Full M0)**
   - [rtl/rgb_frame_writer.sv](rtl/rgb_frame_writer.sv)
   - Packs incoming RGB pixels into 64-bit writes and stores the frame in DDR. Signals `frame_written` when the last row is committed.

3. **Red Mask Datapath**
   - [rtl/red_mask_datapath.sv](rtl/red_mask_datapath.sv)
   - Produces a 1-bit red mask based on channel dominance and thresholding with `min_red_val` and `margin_shift`.

4. **Morphology Filter**
   - [rtl/morphology_filter.sv](rtl/morphology_filter.sv)
   - 3x3 dilation followed by erosion to clean the red mask.

5. **Connected Component Labeling (CCL)**
   - [rtl/ccl_engine.sv](rtl/ccl_engine.sv)
   - Pass 1 labeling: [rtl/ccl_pass1_labeler.sv](rtl/ccl_pass1_labeler.sv)
   - Union-Find compression: [rtl/ccl_uf_resolver.sv](rtl/ccl_uf_resolver.sv)
   - Pass 2 resolved label stream feeds stats.

6. **CCL Stats + Geometry Filter**
   - Stats: [rtl/ccl_stats_collector.sv](rtl/ccl_stats_collector.sv)
   - Geometry: [rtl/geometry_filter.sv](rtl/geometry_filter.sv)
   - Computes area/perimeter/bbox per label, then filters candidates based on size, fill, and aspect constraints.

7. **ROI Fetch (AXI-Full M1)**
   - [rtl/roi_fetcher_axi_master.sv](rtl/roi_fetcher_axi_master.sv)
   - Fetches a 64x64 gray ROI from DDR based on bbox coordinates.

8. **Template Matching**
   - [rtl/template_matching_engine.sv](rtl/template_matching_engine.sv)
   - Binarizes ROI, performs SAD-style matching against template ROM, and selects the best class id and score.

9. **Backend Orchestration**
   - [rtl/backend_processing_unit.sv](rtl/backend_processing_unit.sv)
   - Sequences CCL pass1, union-find resolve, pass2 + stats, geometry filter, ROI fetch, and template matching. Produces `sts_done_flag`, best class, bbox, and a detected-id/bbox-valid pair for the top module.

## Verification Environment
### End-to-End Testbench
- [tb/road_sign_detector_top_tb.sv](tb/road_sign_detector_top_tb.sv)
- Drives the full pipeline from RGB images, compares intermediate stages against golden references, and validates end results.
- Key behaviors:
  - Streams `image_in.hex` into DUT and waits for `irq`.
  - Collects and compares output streams for red mask, morphology, CCL pass1, and geometry filter.
  - Checks AXI protocol stability under backpressure.
  - Dumps RTL outputs into per-frame files under [results/by_image/](results/by_image/).
  - Produces [results/e2e_summary.txt](results/e2e_summary.txt) with pass/fail per frame, detection status, and template classification.

### Unit-Level Testbenches
- Red mask: [tb/tb_red_mask_datapath_image.sv](tb/tb_red_mask_datapath_image.sv)
- Morphology: [tb/tb_morphology_filter_image.sv](tb/tb_morphology_filter_image.sv)
- CCL: [tb/tb_ccl_engine_debug.sv](tb/tb_ccl_engine_debug.sv)
- CCL stats + geometry: [tb/tb_backend_stats_geom.sv](tb/tb_backend_stats_geom.sv)
- ROI + template matching: [tb/tb_roi_template_unit.sv](tb/tb_roi_template_unit.sv)

These unit TBs isolate individual stages, generate actual outputs, and allow comparison against golden files in [data/](data/) and [results/](results/).

## How Validation is Done
- **Mask/Morph/CCL/Geom**: The E2E TB reads golden reference files per frame and counts mismatches. Errors are tracked per stage and summarized in [results/e2e_summary.txt](results/e2e_summary.txt).
- **Template matching**: Best class id and score are compared to golden values from per-frame template reference files. The TB captures full score tables and binarized ROI snapshots for debug.
- **No-bbox cases**: When no geometry-filtered bbox exists, the TB validates that the current id indicates "no sign" and skips template correctness checks.

## E2E Results Summary (Batch Run)
All frames listed below are taken from [results/e2e_summary.txt](results/e2e_summary.txt).

| Frame | Detected | Template | Detection | Result |
| --- | --- | --- | --- | --- |
| 11.png | YES | 100 vel | CORRECT | PASS |
| 12.png | NO | NO | NO_BBOX_NO_SIGN | PASS |
| 13.png | NO | NO | NO_BBOX_NO_SIGN | PASS |
| 14.png | YES | left turn | CORRECT | PASS |
| 15.png | YES | left intersection | CORRECT | PASS |
| 16.png | YES | 100 vel | CORRECT | PASS |
| 17.png | YES | 70 vel | CORRECT | PASS |
| 18.png | YES | 50 vel | CORRECT | PASS |
| 19.png | NO | NO | NO_BBOX_NO_SIGN | PASS |
| 20.png | YES | urban area | CORRECT | PASS |
| 21.png | YES | urban area | CORRECT | PASS |
| 22.png | YES | road works | CORRECT | PASS |
| 23.png | NO | NO | NO_BBOX_NO_SIGN | PASS |
| 24.png | YES | road works | CORRECT | PASS |
| 25.png | YES | 60 vel | CORRECT | PASS |
| 27.png | YES | 60 vel | CORRECT | PASS |
| 6.png | YES | 70 vel | CORRECT | PASS |
| 7.png | YES | no right turn | CORRECT | PASS |
| slippery_road_redcar.jpg | YES | slippery 2 | CORRECT | PASS |
| 2.jpeg | NO | NO | NO_BBOX_NO_SIGN | PASS |
| 5.jpeg | NO | NO | NO_BBOX_NO_SIGN | PASS |

## Per-Frame Visual Results
Each frame below pulls visuals directly from [results/by_image/](results/by_image/) to show the stage-by-stage flow. For NO_BBOX_NO_SIGN frames, template matching visuals are omitted.

### 11.png (Detected: YES, Template: 100 vel)
- ![Red mask](results/by_image/11/red_mask_comparison.png)
- ![Morphology](results/by_image/11/morphology_comparison.png)
- ![CCL resolved](results/by_image/11/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/11/geom_filter_comparison.png)
- ![Template matching](results/by_image/11/template_matching_debug_0.png)

### 12.png (Detected: NO, Template: NO)
- ![Red mask](results/by_image/12/red_mask_comparison.png)
- ![Morphology](results/by_image/12/morphology_comparison.png)
- ![CCL resolved](results/by_image/12/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/12/geom_filter_comparison.png)

### 13.png (Detected: NO, Template: NO)
- ![Red mask](results/by_image/13/red_mask_comparison.png)
- ![Morphology](results/by_image/13/morphology_comparison.png)
- ![CCL resolved](results/by_image/13/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/13/geom_filter_comparison.png)

### 14.png (Detected: YES, Template: left turn)
- ![Red mask](results/by_image/14/red_mask_comparison.png)
- ![Morphology](results/by_image/14/morphology_comparison.png)
- ![CCL resolved](results/by_image/14/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/14/geom_filter_comparison.png)
- ![Template matching](results/by_image/14/template_matching_debug_0.png)

### 15.png (Detected: YES, Template: left intersection)
- ![Red mask](results/by_image/15/red_mask_comparison.png)
- ![Morphology](results/by_image/15/morphology_comparison.png)
- ![CCL resolved](results/by_image/15/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/15/geom_filter_comparison.png)
- ![Template matching](results/by_image/15/template_matching_debug_0.png)

### 16.png (Detected: YES, Template: 100 vel)
- ![Red mask](results/by_image/16/red_mask_comparison.png)
- ![Morphology](results/by_image/16/morphology_comparison.png)
- ![CCL resolved](results/by_image/16/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/16/geom_filter_comparison.png)
- ![Template matching](results/by_image/16/template_matching_debug_0.png)

### 17.png (Detected: YES, Template: 70 vel)
- ![Red mask](results/by_image/17/red_mask_comparison.png)
- ![Morphology](results/by_image/17/morphology_comparison.png)
- ![CCL resolved](results/by_image/17/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/17/geom_filter_comparison.png)
- ![Template matching](results/by_image/17/template_matching_debug_0.png)

### 18.png (Detected: YES, Template: 50 vel)
- ![Red mask](results/by_image/18/red_mask_comparison.png)
- ![Morphology](results/by_image/18/morphology_comparison.png)
- ![CCL resolved](results/by_image/18/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/18/geom_filter_comparison.png)
- ![Template matching](results/by_image/18/template_matching_debug_0.png)

### 19.png (Detected: NO, Template: NO)
- ![Red mask](results/by_image/19/red_mask_comparison.png)
- ![Morphology](results/by_image/19/morphology_comparison.png)
- ![CCL resolved](results/by_image/19/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/19/geom_filter_comparison.png)

### 20.png (Detected: YES, Template: urban area)
- ![Red mask](results/by_image/20/red_mask_comparison.png)
- ![Morphology](results/by_image/20/morphology_comparison.png)
- ![CCL resolved](results/by_image/20/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/20/geom_filter_comparison.png)
- ![Template matching](results/by_image/20/template_matching_debug_0.png)

### 21.png (Detected: YES, Template: urban area)
- ![Red mask](results/by_image/21/red_mask_comparison.png)
- ![Morphology](results/by_image/21/morphology_comparison.png)
- ![CCL resolved](results/by_image/21/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/21/geom_filter_comparison.png)
- ![Template matching](results/by_image/21/template_matching_debug_0.png)

### 22.png (Detected: YES, Template: road works)
- ![Red mask](results/by_image/22/red_mask_comparison.png)
- ![Morphology](results/by_image/22/morphology_comparison.png)
- ![CCL resolved](results/by_image/22/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/22/geom_filter_comparison.png)
- ![Template matching](results/by_image/22/template_matching_debug_0.png)

### 23.png (Detected: NO, Template: NO)
- ![Red mask](results/by_image/23/red_mask_comparison.png)
- ![Morphology](results/by_image/23/morphology_comparison.png)
- ![CCL resolved](results/by_image/23/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/23/geom_filter_comparison.png)

### 24.png (Detected: YES, Template: road works)
- ![Red mask](results/by_image/24/red_mask_comparison.png)
- ![Morphology](results/by_image/24/morphology_comparison.png)
- ![CCL resolved](results/by_image/24/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/24/geom_filter_comparison.png)
- ![Template matching](results/by_image/24/template_matching_debug_0.png)

### 25.png (Detected: YES, Template: 60 vel)
- ![Red mask](results/by_image/25/red_mask_comparison.png)
- ![Morphology](results/by_image/25/morphology_comparison.png)
- ![CCL resolved](results/by_image/25/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/25/geom_filter_comparison.png)
- ![Template matching](results/by_image/25/template_matching_debug_0.png)

### 27.png (Detected: YES, Template: 60 vel)
- ![Red mask](results/by_image/27/red_mask_comparison.png)
- ![Morphology](results/by_image/27/morphology_comparison.png)
- ![CCL resolved](results/by_image/27/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/27/geom_filter_comparison.png)
- ![Template matching](results/by_image/27/template_matching_debug_0.png)

### 6.png (Detected: YES, Template: 70 vel)
- ![Red mask](results/by_image/6/red_mask_comparison.png)
- ![Morphology](results/by_image/6/morphology_comparison.png)
- ![CCL resolved](results/by_image/6/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/6/geom_filter_comparison.png)
- ![Template matching](results/by_image/6/template_matching_debug_0.png)

### 7.png (Detected: YES, Template: no right turn)
- ![Red mask](results/by_image/7/red_mask_comparison.png)
- ![Morphology](results/by_image/7/morphology_comparison.png)
- ![CCL resolved](results/by_image/7/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/7/geom_filter_comparison.png)
- ![Template matching](results/by_image/7/template_matching_debug_0.png)

### slippery_road_redcar.jpg (Detected: YES, Template: slippery 2)
- ![Red mask](results/by_image/slippery_road_redcar/red_mask_comparison.png)
- ![Morphology](results/by_image/slippery_road_redcar/morphology_comparison.png)
- ![CCL resolved](results/by_image/slippery_road_redcar/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/slippery_road_redcar/geom_filter_comparison.png)
- ![Template matching](results/by_image/slippery_road_redcar/template_matching_debug_0.png)

### 2.jpeg (Detected: NO, Template: NO)
- ![Red mask](results/by_image/2/red_mask_comparison.png)
- ![Morphology](results/by_image/2/morphology_comparison.png)
- ![CCL resolved](results/by_image/2/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/2/geom_filter_comparison.png)

### 5.jpeg (Detected: NO, Template: NO)
- ![Red mask](results/by_image/5/red_mask_comparison.png)
- ![Morphology](results/by_image/5/morphology_comparison.png)
- ![CCL resolved](results/by_image/5/ccl_resolved_vis.png)
- ![Geometry filter](results/by_image/5/geom_filter_comparison.png)
