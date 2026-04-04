#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
from PIL import Image


VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_rgb(path):
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def detect_red_border(rgb, red_min, red_diff):
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r >= red_min) & (r >= g + red_diff) & (r >= b + red_diff)


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        y0 = max(0, dy)
        y1 = h + min(0, dy)
        for dx in range(-radius, radius + 1):
            x0 = max(0, dx)
            x1 = w + min(0, dx)
            out[y0:y1, x0:x1] |= mask[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def flood_fill_outside(block_mask):
    h, w = block_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue = []

    for x in range(w):
        if not block_mask[0, x]:
            outside[0, x] = True
            queue.append((0, x))
        if not block_mask[h - 1, x]:
            outside[h - 1, x] = True
            queue.append((h - 1, x))
    for y in range(h):
        if not block_mask[y, 0]:
            outside[y, 0] = True
            queue.append((y, 0))
        if not block_mask[y, w - 1]:
            outside[y, w - 1] = True
            queue.append((y, w - 1))

    head = 0
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while head < len(queue):
        y, x = queue[head]
        head += 1
        for dy, dx in nbrs:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < h and 0 <= xx < w:
                if not block_mask[yy, xx] and not outside[yy, xx]:
                    outside[yy, xx] = True
                    queue.append((yy, xx))

    return outside


def whiten_outside_red_border(rgb, red_min, red_diff, dilate_radius, min_red_pixels):
    red_mask = detect_red_border(rgb, red_min, red_diff)
    if int(red_mask.sum()) < min_red_pixels:
        return rgb, False
    border = dilate_mask(red_mask, dilate_radius)
    outside = flood_fill_outside(border)
    out = rgb.copy()
    out[outside] = 255
    return out, True


def process_dir(in_dir, out_dir, red_min, red_diff, dilate_radius, min_red_pixels, in_place):
    if not os.path.isdir(in_dir):
        raise RuntimeError(f"Input directory not found: {in_dir}")

    if in_place:
        out_dir = in_dir
    else:
        os.makedirs(out_dir, exist_ok=True)

    count = 0
    for fname in sorted(os.listdir(in_dir)):
        if not fname.lower().endswith(VALID_EXT):
            continue
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)

        rgb = load_rgb(in_path)
        out, ok = whiten_outside_red_border(rgb, red_min, red_diff, dilate_radius, min_red_pixels)
        if not ok:
            print(f"Warning: red border not detected for {fname}, leaving unchanged")

        Image.fromarray(out, mode="RGB").save(out_path)
        count += 1

    print(f"Processed {count} templates")
    print(f"Output directory: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Whiten template backgrounds outside red border")
    parser.add_argument("--in", dest="in_dir", required=True, help="Input templates directory")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory (default: <in>_white)")
    parser.add_argument("--red-min", type=int, default=120, help="Minimum red channel value")
    parser.add_argument("--red-diff", type=int, default=40, help="Minimum red dominance over G/B")
    parser.add_argument("--dilate", type=int, default=1, help="Border dilation radius")
    parser.add_argument("--min-red", type=int, default=20, help="Minimum red pixels to accept detection")
    parser.add_argument("--in-place", action="store_true", help="Overwrite templates in place")
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.in_dir.rstrip("/") + "_white"

    process_dir(args.in_dir, out_dir, args.red_min, args.red_diff, args.dilate, args.min_red, args.in_place)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
