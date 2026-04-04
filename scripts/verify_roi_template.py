import os
import sys
import numpy as np
from PIL import Image

NO_SIGN_SCORE_THRESHOLD = 230

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def load_lines_ints(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            data.append([int(tok) for tok in s.split()])
    return data


def parse_template_mapping(map_path):
    entries = []
    if not map_path or not os.path.exists(map_path):
        return entries
    with open(map_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if not s.startswith("ID") or ":" not in s:
                continue
            left, right = s.split(":", 1)
            left_toks = left.strip().split()
            if len(left_toks) < 2:
                continue
            try:
                idx = int(left_toks[1])
            except ValueError:
                continue
            right = right.strip()
            if not right:
                continue
            fname, sep, label = right.partition(" - ")
            fname = fname.strip()
            label = label.strip() if sep else os.path.splitext(fname)[0]
            entries.append((idx, fname, label))
    entries.sort(key=lambda x: x[0])
    return entries


def load_bin_rows_hex(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(int(s, 16))
            except ValueError:
                continue
    return rows


def load_rtl_scores(path, count):
    if not path or not os.path.exists(path):
        return None
    scores = [None] * count
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            toks = s.split()
            if len(toks) < 2:
                continue
            try:
                tid = int(toks[0])
                sc = int(toks[1])
            except ValueError:
                continue
            if 0 <= tid < count:
                scores[tid] = sc
    if any(v is None for v in scores):
        return None
    return scores


def bin_rows_to_array(rows):
    arr = np.zeros((32, 32), dtype=np.uint8)
    if not rows:
        return arr
    for y in range(min(32, len(rows))):
        val = rows[y]
        for x in range(32):
            if val & (1 << (31 - x)):
                arr[y, x] = 1
    return arr


def load_golden(path):
    rows = []
    for toks in load_lines_ints(path):
        if len(toks) < 7:
            continue
        rows.append({
            "roi_id": toks[0],
            "xmin": toks[1],
            "xmax": toks[2],
            "ymin": toks[3],
            "ymax": toks[4],
            "best_id": toks[5],
            "best_score": toks[6],
        })
    return rows


def load_actual(path):
    rows = []
    for toks in load_lines_ints(path):
        if len(toks) >= 7:
            rows.append({
                "roi_id": toks[0],
                "best_id": toks[5],
                "best_score": toks[6],
            })
        elif len(toks) >= 3:
            rows.append({
                "roi_id": toks[0],
                "best_id": toks[1],
                "best_score": toks[2],
            })
    return rows


def apply_no_sign_threshold(rows, threshold):
    out = []
    for row in rows:
        row_copy = dict(row)
        if row_copy.get("best_score") is not None and row_copy["best_score"] >= threshold:
            row_copy["best_id"] = -1
            row_copy["best_score"] = threshold
        out.append(row_copy)
    return out


def rgb_to_gray_u8(img_rgb_u8):
    r = img_rgb_u8[:, :, 0].astype(np.uint16)
    g = img_rgb_u8[:, :, 1].astype(np.uint16)
    b = img_rgb_u8[:, :, 2].astype(np.uint16)
    gray = (77 * r + 150 * g + 29 * b) >> 8
    return gray.astype(np.uint8)


def resize_image_fhd(img_rgb_u8):
    if img_rgb_u8.shape[0] == 1080 and img_rgb_u8.shape[1] == 1920:
        return img_rgb_u8
    img = Image.fromarray(img_rgb_u8, mode="RGB")
    img = img.resize((1920, 1080), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def downsample_roi_gray(img_rgb_u8, xmin, xmax, ymin, ymax):
    roi_w = xmax - xmin + 1
    roi_h = ymax - ymin + 1
    out = np.zeros((64, 64), dtype=np.uint8)

    for y_dst in range(64):
        y_src = ymin + ((y_dst * roi_h) >> 6)
        for x_dst in range(64):
            x_src = xmin + ((x_dst * roi_w) >> 6)
            r, g, b = img_rgb_u8[y_src, x_src]
            gray = (77 * int(r) + 150 * int(g) + 29 * int(b)) >> 8
            out[y_dst, x_dst] = gray

    return out


def binarize_center_legacy(gray64):
    center = gray64[16:48, 16:48]
    mean_val = int(np.mean(center)) if center.size else 0
    thr = mean_val - 15
    if thr < 0:
        thr = 0
    bin_center = (center < thr).astype(np.uint8)
    return bin_center


def binarize_center_template(gray64):
    center = gray64[16:48, 16:48]
    mean_val = int(np.mean(center)) if center.size else 0
    return (center < mean_val).astype(np.uint8)


def pack_bin_rows(bin32):
    rows = []
    for y in range(32):
        val = 0
        for x in range(32):
            if bin32[y, x]:
                val |= 1 << (31 - x)
        rows.append(val)
    return rows


def popcount32(x):
    return int(bin(x & 0xFFFFFFFF).count("1"))


def score_template_with_shift(bin_rows, tmpl_rows):
    best = None
    best_dx = 0
    best_dy = 0

    for dy in range(-4, 5):
        for dx in range(-4, 5):
            mismatches = 0
            for y in range(32):
                yy = y + dy
                if yy < 0 or yy >= 32:
                    continue
                row = bin_rows[yy]
                if dx > 0:
                    shifted = (row << dx) & 0xFFFFFFFF
                    mask = (0xFFFFFFFF << dx) & 0xFFFFFFFF
                elif dx < 0:
                    shifted = (row >> (-dx)) & 0xFFFFFFFF
                    mask = (0xFFFFFFFF >> (-dx)) & 0xFFFFFFFF
                else:
                    shifted = row
                    mask = 0xFFFFFFFF

                xor_val = (shifted ^ tmpl_rows[y]) & mask
                mismatches += popcount32(xor_val)
            if best is None or mismatches < best:
                best = mismatches
                best_dx = dx
                best_dy = dy

    return best if best is not None else 0, best_dx, best_dy


def build_xor_map(bin_center, tmpl_bin, dx, dy):
    out = np.zeros((32, 32), dtype=np.uint8)
    for y in range(32):
        yy = y + dy
        if yy < 0 or yy >= 32:
            continue
        for x in range(32):
            xx = x + dx
            if xx < 0 or xx >= 32:
                continue
            a = bin_center[yy, xx]
            b = tmpl_bin[y, x]
            out[y, x] = 255 if (a ^ b) else 0
    return out


def build_heatmap_strip(xor_maps, pad=2):
    if not xor_maps:
        return None
    h, w = xor_maps[0].shape
    strip_w = (w * len(xor_maps)) + (pad * (len(xor_maps) - 1))
    strip = np.zeros((h, strip_w), dtype=np.uint8)
    x = 0
    for m in xor_maps:
        strip[:, x:x + w] = m
        x += w + pad
    return strip


def label_for_id(id_to_label, tid):
    if id_to_label and tid in id_to_label:
        return id_to_label[tid]
    return f"id{tid}"


def load_templates_gray(template_dir, out_h=64, out_w=64, map_entries=None):
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if map_entries:
        files = [entry[1] for entry in map_entries]
    else:
        files = [f for f in sorted(os.listdir(template_dir)) if f.lower().endswith(valid_ext)]
    if not files:
        raise RuntimeError(f"No templates found in {template_dir}")

    templates = []
    for fname in files:
        if not fname.lower().endswith(valid_ext):
            raise RuntimeError(f"Template file has invalid extension: {fname}")
        path = os.path.join(template_dir, fname)
        if not os.path.exists(path):
            raise RuntimeError(f"Template file missing: {path}")
        img = Image.open(path).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        gray = rgb_to_gray_u8(img_np)
        y_idx = (np.arange(out_h) * gray.shape[0]) // out_h
        x_idx = (np.arange(out_w) * gray.shape[1]) // out_w
        gray64 = gray[y_idx[:, None], x_idx[None, :]]
        templates.append((fname, gray64))
    return templates



def resolve_stage_file(image_path, project_dir, filename):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    by_image = os.path.join(project_dir, "results", "by_image", stem, filename)
    if os.path.exists(by_image):
        return by_image
    fallback = os.path.join(project_dir, "data", filename)
    return fallback


def load_roi_list(project_dir):
    path = os.path.join(project_dir, "data", "roi_list.txt")
    roi_map = {}
    if not os.path.exists(path):
        return roi_map
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 6:
                continue
            roi_id = int(toks[0])
            label = int(toks[5])
            roi_map[roi_id] = label
    return roi_map


def load_binary_mask(path, shape):
    data = np.loadtxt(path, dtype=np.uint8)
    if data.size != shape[0] * shape[1]:
        raise RuntimeError(f"Mask size mismatch for {path}")
    return data.reshape(shape)


def draw_bbox(ax, xmin, xmax, ymin, ymax, color="yellow"):
    ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color=color, linewidth=2)


def visualize(image_path, template_dir, golden_rows, out_dir):
    if not HAS_MPL:
        print("Visualization skipped: matplotlib not available")
        return
    img = Image.open(image_path).convert("RGB")
    img_np = resize_image_fhd(np.array(img, dtype=np.uint8))

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    mask_path = resolve_stage_file(image_path, project_dir, "mask_out.txt")
    morph_path = resolve_stage_file(image_path, project_dir, "morph_out.txt")
    bbox_path = resolve_stage_file(image_path, project_dir, "geom_bboxes_golden.txt")

    if not os.path.exists(mask_path) or not os.path.exists(morph_path) or not os.path.exists(bbox_path):
        print("Skipping visualization: missing mask/morph/bbox inputs")
        return

    mask = load_binary_mask(mask_path, (1080, 1920))
    morph = load_binary_mask(morph_path, (1080, 1920))

    bboxes = []
    for toks in load_lines_ints(bbox_path):
        if len(toks) >= 5:
            bboxes.append((toks[1], toks[2], toks[3], toks[4]))

    map_path = os.path.join(project_dir, "data", "template_mapping.txt")
    map_entries = parse_template_mapping(map_path)
    id_to_label = {idx: label for idx, _fname, label in map_entries}

    templates = load_templates_gray(template_dir, out_h=64, out_w=64, map_entries=map_entries if map_entries else None)
    template_bins = [binarize_center_template(g) for _name, g in templates]
    template_rows = [pack_bin_rows(tbin) for tbin in template_bins]

    os.makedirs(out_dir, exist_ok=True)

    actual_top3_path = resolve_stage_file(image_path, project_dir, "actual_top3.txt")
    actual_top3 = {}
    if os.path.exists(actual_top3_path):
        with open(actual_top3_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                toks = s.split()
                if len(toks) >= 7:
                    rid = int(toks[0])
                    actual_top3[rid] = toks[1:7]

    rtl_bin_path = resolve_stage_file(image_path, project_dir, "actual_roi_bin_0.txt")
    rtl_bin_rows = load_bin_rows_hex(rtl_bin_path)
    rtl_scores_path = resolve_stage_file(image_path, project_dir, "actual_scores_full.txt")
    rtl_scores_file = load_rtl_scores(rtl_scores_path, len(template_rows))

    for row in golden_rows[:3]:
        xmin = row["xmin"]
        xmax = row["xmax"]
        ymin = row["ymin"]
        ymax = row["ymax"]
        best_id = row["best_id"]

        roi_gray = downsample_roi_gray(img_np, xmin, xmax, ymin, ymax)
        roi_bin = binarize_center_legacy(roi_gray)
        bin_rows = pack_bin_rows(roi_bin)

        scores = []
        best_dx_by_id = []
        best_dy_by_id = []
        for tidx, tmpl_rows in enumerate(template_rows):
            score, dx, dy = score_template_with_shift(bin_rows, tmpl_rows)
            scores.append(score)
            best_dx_by_id.append(dx)
            best_dy_by_id.append(dy)

        rtl_bin_rows_use = None
        rtl_bin_center = None
        rtl_bin_img = None
        if row["roi_id"] == 0 and len(rtl_bin_rows) >= 32:
            rtl_bin_rows_use = rtl_bin_rows[:32]
            rtl_bin_center = bin_rows_to_array(rtl_bin_rows_use)
            rtl_bin_img = (rtl_bin_center > 0).astype(np.uint8)

        rtl_scores = None
        rtl_best_dx_by_id = None
        rtl_best_dy_by_id = None
        if rtl_scores_file is not None:
            rtl_scores = rtl_scores_file
        if rtl_bin_rows_use is not None:
            rtl_best_dx_by_id = []
            rtl_best_dy_by_id = []
            for tmpl_rows in template_rows:
                _r_score, r_dx, r_dy = score_template_with_shift(rtl_bin_rows_use, tmpl_rows)
                rtl_best_dx_by_id.append(r_dx)
                rtl_best_dy_by_id.append(r_dy)
            if rtl_scores is None:
                rtl_scores = []
                for tmpl_rows in template_rows:
                    r_score, _r_dx, _r_dy = score_template_with_shift(rtl_bin_rows_use, tmpl_rows)
                    rtl_scores.append(r_score)

        score_pairs = list(enumerate(scores))
        score_pairs.sort(key=lambda x: x[1])
        top3 = score_pairs[:3]

        tmpl_bin = template_bins[best_id]
        xor_map = build_xor_map(roi_bin, tmpl_bin, best_dx_by_id[best_id], best_dy_by_id[best_id])

        golden_top3_maps = []
        for tid, _score in top3:
            golden_top3_maps.append(build_xor_map(roi_bin, template_bins[tid], best_dx_by_id[tid], best_dy_by_id[tid]))

        rtl_top_pairs = []
        rtl_top_maps = []
        rtl_best_id = None
        rtl_best_map = None

        if rtl_scores is not None:
            rtl_score_pairs = list(enumerate(rtl_scores))
            rtl_score_pairs.sort(key=lambda x: x[1])
            rtl_top_pairs = rtl_score_pairs[:3]
            if rtl_top_pairs:
                rtl_best_id = rtl_top_pairs[0][0]
                rtl_best_map = build_xor_map(
                    rtl_bin_center,
                    template_bins[rtl_best_id],
                    rtl_best_dx_by_id[rtl_best_id],
                    rtl_best_dy_by_id[rtl_best_id]
                )
            for tid, _sc in rtl_top_pairs:
                rtl_top_maps.append(build_xor_map(
                    rtl_bin_center,
                    template_bins[tid],
                    rtl_best_dx_by_id[tid],
                    rtl_best_dy_by_id[tid]
                ))
        else:
            rtl_top = actual_top3.get(row["roi_id"])
            if rtl_top is not None:
                rtl_ids = [int(rtl_top[0]), int(rtl_top[2]), int(rtl_top[4])]
                rtl_scores_local = [int(rtl_top[1]), int(rtl_top[3]), int(rtl_top[5])]
                for tid, sc in zip(rtl_ids, rtl_scores_local):
                    if 0 <= tid < len(template_bins):
                        rtl_top_pairs.append((tid, sc))
                        rtl_top_maps.append(build_xor_map(roi_bin, template_bins[tid], best_dx_by_id[tid], best_dy_by_id[tid]))

        fig = plt.figure(figsize=(18, 16))
        grid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(grid[0, 0])
        ax0.imshow(img_np)
        draw_bbox(ax0, xmin, xmax, ymin, ymax)
        ax0.set_title("Original + ROI")
        ax0.axis("off")

        ax1 = fig.add_subplot(grid[0, 1])
        ax1.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax1.set_title("Red Mask")
        ax1.axis("off")

        ax2 = fig.add_subplot(grid[0, 2])
        ax2.imshow(morph, cmap="gray", vmin=0, vmax=1)
        for bx in bboxes:
            draw_bbox(ax2, bx[0], bx[1], bx[2], bx[3], color="lime")
        ax2.set_title("Morph + CCL BBoxes")
        ax2.axis("off")

        ax3 = fig.add_subplot(grid[0, 3])
        ax3.axis("off")
        ax3.set_title("Template Matching")

        ax4 = fig.add_subplot(grid[1, 0])
        ax4.imshow(roi_gray, cmap="gray", vmin=0, vmax=255)
        ax4.set_title("Extracted ROI")
        ax4.axis("off")

        ax5 = fig.add_subplot(grid[1, 1])
        ax5.imshow(roi_bin, cmap="gray", vmin=0, vmax=1)
        ax5.set_title("Normalized ROI")
        ax5.axis("off")

        ax6 = fig.add_subplot(grid[1, 2])
        ax6.imshow(xor_map, cmap="magma", vmin=0, vmax=255)
        ax6.set_title("SAD Heatmap")
        ax6.axis("off")

        ax7 = fig.add_subplot(grid[1, 3])
        ax7.bar(range(len(scores)), scores, color="skyblue")
        ax7.bar(best_id, scores[best_id], color="limegreen")
        ax7.set_title("Golden Scores per Template")
        ax7.set_xlabel("Template ID")
        ax7.set_ylabel("Mismatch Count")

        top3_text = "Golden top3: " + ", ".join([
            f"{tid}({label_for_id(id_to_label, tid)}:{sc})" for tid, sc in top3
        ])
        ax7.text(0.02, 0.98, top3_text, transform=ax7.transAxes,
                 fontsize=9, va="top", ha="left")
        if rtl_top_pairs:
            rtl_text = "RTL top3: " + ", ".join([
                f"{tid}({label_for_id(id_to_label, tid)}:{sc})" for tid, sc in rtl_top_pairs
            ])
            ax7.text(0.02, 0.90, rtl_text, transform=ax7.transAxes,
                     fontsize=9, va="top", ha="left")

        ax8 = fig.add_subplot(grid[2, 0])
        ax8.imshow(tmpl_bin, cmap="gray", vmin=0, vmax=1)
        best_label = label_for_id(id_to_label, best_id)
        ax8.set_title(f"Best Template (ID={best_id}, {best_label})")
        ax8.axis("off")

        ax9 = fig.add_subplot(grid[2, 1])
        ax9.imshow(img_np[ymin:ymax + 1, xmin:xmax + 1])
        ax9.set_title("ROI (Color)")
        ax9.axis("off")

        ax10 = fig.add_subplot(grid[2, 2])
        ax10.imshow(morph[ymin:ymax + 1, xmin:xmax + 1], cmap="gray", vmin=0, vmax=1)
        ax10.set_title("ROI Morph")
        ax10.axis("off")

        ax12 = fig.add_subplot(grid[3, 0])
        if rtl_bin_img is not None:
            ax12.imshow(rtl_bin_img, cmap="gray", vmin=0, vmax=1)
        else:
            ax12.imshow(np.zeros((32, 32), dtype=np.uint8), cmap="gray", vmin=0, vmax=1)
        ax12.set_title("RTL Normalized ROI")
        ax12.axis("off")

        ax13 = fig.add_subplot(grid[3, 1])
        if rtl_scores is not None:
            ax13.bar(range(len(rtl_scores)), rtl_scores, color="lightcoral")
            if rtl_best_id is not None:
                ax13.bar(rtl_best_id, rtl_scores[rtl_best_id], color="crimson")
            ax13.set_title("RTL Scores per Template")
            ax13.set_xlabel("Template ID")
            ax13.set_ylabel("Mismatch Count")
            if rtl_top_pairs:
                rtl_text = "RTL top3: " + ", ".join([
                    f"{tid}({label_for_id(id_to_label, tid)}:{sc})" for tid, sc in rtl_top_pairs
                ])
                ax13.text(0.02, 0.98, rtl_text, transform=ax13.transAxes,
                          fontsize=9, va="top", ha="left")
        else:
            ax13.axis("off")
            ax13.set_title("RTL Scores per Template")
            ax13.text(0.5, 0.5, "RTL scores unavailable", ha="center", va="center")

        ax14 = fig.add_subplot(grid[3, 2])
        if rtl_best_map is not None:
            ax14.imshow(rtl_best_map, cmap="magma", vmin=0, vmax=255)
        else:
            ax14.imshow(np.zeros((32, 32), dtype=np.uint8), cmap="magma", vmin=0, vmax=255)
        ax14.set_title("RTL Best Heatmap")
        ax14.axis("off")

        ax15 = fig.add_subplot(grid[3, 3])
        if rtl_best_id is not None:
            ax15.imshow(template_bins[rtl_best_id], cmap="gray", vmin=0, vmax=1)
        else:
            ax15.imshow(np.zeros((32, 32), dtype=np.uint8), cmap="gray", vmin=0, vmax=1)
        ax15.set_title("RTL Best Template")
        ax15.axis("off")

        golden_strip = build_heatmap_strip(golden_top3_maps)
        rtl_strip = build_heatmap_strip(rtl_top_maps)

        ax16 = fig.add_subplot(grid[4, 0:2])
        if golden_strip is not None:
            ax16.imshow(golden_strip, cmap="magma", vmin=0, vmax=255)
        ax16.set_title("Golden top3 heatmaps")
        ax16.axis("off")

        ax17 = fig.add_subplot(grid[4, 2:4])
        if rtl_strip is not None:
            ax17.imshow(rtl_strip, cmap="magma", vmin=0, vmax=255)
        ax17.set_title("RTL top3 heatmaps")
        ax17.axis("off")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"template_matching_debug_{row['roi_id']}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Visualization: {out_path}")


def main(golden_path, actual_path, out_dir, image_path, template_dir):
    golden_rows_raw = load_golden(golden_path)
    actual_rows_raw = load_actual(actual_path)

    golden_rows = apply_no_sign_threshold(golden_rows_raw, NO_SIGN_SCORE_THRESHOLD)
    actual_rows = apply_no_sign_threshold(actual_rows_raw, NO_SIGN_SCORE_THRESHOLD)

    actual_by_id = {r["roi_id"]: r for r in actual_rows}

    class_mismatch = []
    score_mismatch = []

    for g in golden_rows:
        r = actual_by_id.get(g["roi_id"])
        if r is None:
            class_mismatch.append((g["roi_id"], "missing"))
            continue
        if r["best_id"] != g["best_id"]:
            class_mismatch.append((g["roi_id"], g["best_id"], r["best_id"]))
        if r["best_score"] != g["best_score"]:
            score_mismatch.append((g["roi_id"], g["best_score"], r["best_score"]))

    status = "PASS" if not class_mismatch and not score_mismatch else "FAIL"

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "template_matching_verify.txt")
    with open(report_path, "w") as f:
        f.write("STAGE: TEMPLATE_MATCHING\n")
        f.write(f"GOLDEN_ROIS: {len(golden_rows)}\n")
        f.write(f"ACTUAL_ROIS: {len(actual_rows)}\n")
        f.write(f"CLASS_MISMATCHES: {len(class_mismatch)}\n")
        f.write(f"SCORE_MISMATCHES: {len(score_mismatch)}\n")
        f.write(f"STATUS: {status}\n")

        if class_mismatch:
            f.write("\nClass mismatches (roi_id golden actual):\n")
            for row in class_mismatch[:200]:
                f.write("{}\n".format(" ".join(str(x) for x in row)))
        if score_mismatch:
            f.write("\nScore mismatches (roi_id golden actual):\n")
            for row in score_mismatch[:200]:
                f.write("{}\n".format(" ".join(str(x) for x in row)))

    print(f"Report: {report_path}")
    print(f"Status: {status}")

    visualize(image_path, template_dir, golden_rows_raw, out_dir)

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    no_exit = False
    args = []
    for a in sys.argv[1:]:
        if a == "--no-exit":
            no_exit = True
            continue
        args.append(a)
    if len(args) != 5:
        print("Usage: python verify_roi_template.py <golden.txt> <actual.txt> <out_dir> <image_path> <template_dir>")
        sys.exit(1)

    main(args[0], args[1], args[2], args[3], args[4])
    if no_exit:
        sys.exit(0)
