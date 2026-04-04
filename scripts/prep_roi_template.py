import os
import sys
import numpy as np
from PIL import Image

NO_SIGN_SCORE_THRESHOLD = 230


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


def write_image_hex(img_rgb_u8, out_path):
    h, w, _ = img_rgb_u8.shape
    with open(out_path, "w") as f:
        for y in range(h):
            for x in range(w):
                r, g, b = img_rgb_u8[y, x]
                f.write(f"{r:02x}{g:02x}{b:02x}\n")


def resolve_bbox_path(image_path, bbox_path, project_dir):
    if bbox_path:
        return bbox_path
    stem = os.path.splitext(os.path.basename(image_path))[0]
    by_image = os.path.join(project_dir, "results", "by_image", stem, "geom_bboxes_golden.txt")
    if os.path.exists(by_image):
        return by_image
    fallback = os.path.join(project_dir, "data", "geom_bboxes_golden.txt")
    return fallback

def load_bboxes(path):
    bboxes = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 5:
                continue
            label = int(toks[0])
            xmin = int(toks[1])
            xmax = int(toks[2])
            ymin = int(toks[3])
            ymax = int(toks[4])
            bboxes.append((label, xmin, xmax, ymin, ymax))
    return bboxes


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


def pack_bin_rows(bin32):
    rows = []
    for y in range(32):
        val = 0
        for x in range(32):
            if bin32[y, x]:
                val |= 1 << (31 - x)
        rows.append(val)
    return rows


def load_templates_binary(template_dir, map_path=None):
    from prep_templates_mem import load_templates_gray, binarize_center, pack_bin_rows, parse_template_mapping

    map_entries = parse_template_mapping(map_path) if map_path else []
    templates_gray = load_templates_gray(template_dir, out_h=64, out_w=64, map_entries=map_entries if map_entries else None)
    templates_bin = []
    for fname, gray64 in templates_gray:
        bin_center = binarize_center(gray64)
        rows = pack_bin_rows(bin_center)
        templates_bin.append((fname, rows))
    return templates_bin


def popcount32(x):
    return int(bin(x & 0xFFFFFFFF).count("1"))


def score_template(bin_rows, tmpl_rows):
    best = None
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
    return best if best is not None else 0


def main(image_path, out_dir, bbox_path=None, template_dir=None):
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if template_dir is None:
        template_dir_default = os.path.join(project_dir, "pyton", "Templates_white")
        if os.path.isdir(template_dir_default):
            template_dir = template_dir_default
        else:
            template_dir = os.path.join(project_dir, "pyton", "Templates")

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    img_np = resize_image_fhd(img_np)

    os.makedirs(out_dir, exist_ok=True)
    image_hex_path = os.path.join(out_dir, "image_in.hex")
    write_image_hex(img_np, image_hex_path)

    bbox_path = resolve_bbox_path(image_path, bbox_path, project_dir)
    bboxes = load_bboxes(bbox_path)
    if not bboxes:
        print(f"No bboxes found in: {bbox_path}")
        sys.exit(1)

    print("Legacy mode: mean-15 threshold")

    roi_list_path = os.path.join(out_dir, "roi_list.txt")
    with open(roi_list_path, "w") as f:
        f.write("# roi_id xmin xmax ymin ymax label\n")
        for idx, (label, xmin, xmax, ymin, ymax) in enumerate(bboxes):
            f.write(f"{idx} {xmin} {xmax} {ymin} {ymax} {label}\n")

    map_path = os.path.join(project_dir, "data", "template_mapping.txt")
    templates_bin = load_templates_binary(template_dir, map_path)

    golden_path = os.path.join(out_dir, "template_matching_golden.txt")
    golden_bin_path = os.path.join(out_dir, "golden_roi_bin_0.txt")
    golden_bin_rows = None
    with open(golden_path, "w") as f:
        f.write("# roi_id xmin xmax ymin ymax best_class_id best_score\n")
        for idx, (label, xmin, xmax, ymin, ymax) in enumerate(bboxes):
            roi_gray = downsample_roi_gray(img_np, xmin, xmax, ymin, ymax)
            bin_center = binarize_center_legacy(roi_gray)
            bin_rows = pack_bin_rows(bin_center)
            if idx == 0:
                golden_bin_rows = bin_rows

            best_id = 0
            best_score = None
            for tidx, (_fname, tmpl_rows) in enumerate(templates_bin):
                score = score_template(bin_rows, tmpl_rows)
                if best_score is None or score < best_score:
                    best_score = score
                    best_id = tidx

            if best_score is not None and best_score >= NO_SIGN_SCORE_THRESHOLD:
                best_id = -1
                best_score = NO_SIGN_SCORE_THRESHOLD

            f.write(f"{idx} {xmin} {xmax} {ymin} {ymax} {best_id} {best_score}\n")

    if golden_bin_rows is not None:
        with open(golden_bin_path, "w") as f:
            for val in golden_bin_rows:
                f.write(f"{val:08x}\n")

    print(f"Image hex: {image_hex_path}")
    print(f"ROI list: {roi_list_path}")
    print(f"Golden results: {golden_path}")
    if golden_bin_rows is not None:
        print(f"Golden bin rows: {golden_bin_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2 or len(args) > 4:
        print("Usage: python prep_roi_template.py <image_path> <out_dir> [bbox_path] [template_dir]")
        sys.exit(1)

    image_path = args[0]
    out_dir = args[1]
    bbox_path = args[2] if len(args) >= 3 else None
    template_dir = args[3] if len(args) == 4 else None

    main(image_path, out_dir, bbox_path, template_dir)
