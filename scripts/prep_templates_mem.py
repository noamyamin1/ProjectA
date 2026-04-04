import os
import sys
import numpy as np
from PIL import Image


def rgb_to_gray_u8(img_rgb_u8):
    r = img_rgb_u8[:, :, 0].astype(np.uint16)
    g = img_rgb_u8[:, :, 1].astype(np.uint16)
    b = img_rgb_u8[:, :, 2].astype(np.uint16)
    gray = (77 * r + 150 * g + 29 * b) >> 8
    return gray.astype(np.uint8)


def resize_nn_gray_u8(img_u8, out_h, out_w):
    in_h, in_w = img_u8.shape
    y_idx = (np.arange(out_h) * in_h) // out_h
    x_idx = (np.arange(out_w) * in_w) // out_w
    return img_u8[y_idx[:, None], x_idx[None, :]]


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
        gray64 = resize_nn_gray_u8(gray, out_h, out_w)
        templates.append((fname, gray64))
    return templates


def binarize_center(gray64):
    center = gray64[16:48, 16:48]
    if center.size == 0:
        mean_val = 0
    else:
        mean_val = int(np.mean(center))
    bin_center = (center < mean_val).astype(np.uint8)
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


def write_templates_mem(templates, out_mem_path, out_map_path, map_entries=None):
    mem_lines = []
    for idx, (fname, gray64) in enumerate(templates):
        bin_center = binarize_center(gray64)
        mem_lines.extend(pack_bin_rows(bin_center))

    map_lines = []
    if map_entries:
        if len(map_entries) != len(templates):
            raise RuntimeError("Mapping entries do not match template count")
        for idx, (map_id, fname, label) in enumerate(map_entries):
            if map_id != idx:
                raise RuntimeError("Mapping IDs must be contiguous from 0")
            if templates[idx][0] != fname:
                raise RuntimeError("Mapping order does not match template list")
            map_lines.append(f"ID {map_id} : {fname} - {label}")
    else:
        for idx, (fname, _gray64) in enumerate(templates):
            label = os.path.splitext(fname)[0]
            map_lines.append(f"ID {idx} : {fname} - {label}")

    with open(out_mem_path, "w") as f:
        for val in mem_lines:
            f.write(f"{val:08x}\n")

    with open(out_map_path, "w") as f:
        for line in map_lines:
            f.write(line + "\n")

    print(f"Wrote templates.mem: {out_mem_path}")
    print(f"Wrote template mapping: {out_map_path}")
    print(f"Templates: {len(templates)} | Lines: {len(mem_lines)}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python prep_templates_mem.py <template_dir> <out_mem_path> [out_map_path]")
        sys.exit(1)

    template_dir = sys.argv[1]
    out_mem_path = sys.argv[2]
    out_map_path = sys.argv[3] if len(sys.argv) == 4 else os.path.join(os.path.dirname(out_mem_path), "template_mapping.txt")

    map_entries = parse_template_mapping(out_map_path)
    templates = load_templates_gray(template_dir, out_h=64, out_w=64, map_entries=map_entries if map_entries else None)
    write_templates_mem(templates, out_mem_path, out_map_path, map_entries if map_entries else None)
