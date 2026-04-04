#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
from PIL import Image


def resize_image_fhd(img_rgb_u8):
    if img_rgb_u8.shape[0] == 1080 and img_rgb_u8.shape[1] == 1920:
        return img_rgb_u8
    img = Image.fromarray(img_rgb_u8, mode="RGB")
    img = img.resize((1920, 1080), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def resolve_roi_list(image_path, project_dir, roi_list_override=None):
    if roi_list_override:
        return roi_list_override
    stem = os.path.splitext(os.path.basename(image_path))[0]
    by_image = os.path.join(project_dir, "results", "by_image", stem, "roi_list.txt")
    if os.path.exists(by_image):
        return by_image
    return os.path.join(project_dir, "data", "roi_list.txt")


def load_first_roi(roi_list_path):
    if not os.path.exists(roi_list_path):
        raise FileNotFoundError(roi_list_path)
    with open(roi_list_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 5:
                continue
            xmin = int(toks[1])
            xmax = int(toks[2])
            ymin = int(toks[3])
            ymax = int(toks[4])
            return xmin, xmax, ymin, ymax
    raise RuntimeError(f"No ROI entries in {roi_list_path}")


def extract_roi_gray(image_path, roi_list_path):
    img = Image.open(image_path).convert("RGB")
    img_np = resize_image_fhd(np.array(img, dtype=np.uint8))
    xmin, xmax, ymin, ymax = load_first_roi(roi_list_path)
    roi = img_np[ymin:ymax + 1, xmin:xmax + 1]
    roi_img = Image.fromarray(roi, mode="RGB")
    roi_img = roi_img.resize((64, 64), Image.BILINEAR)
    roi_gray = roi_img.convert("L")
    return np.array(roi_gray, dtype=np.uint8)


def binarize_center_legacy(gray64):
    center = gray64[16:48, 16:48]
    mean_val = int(np.mean(center)) if center.size else 0
    thr = mean_val - 15
    if thr < 0:
        thr = 0
    return (center < thr).astype(np.uint8)


def build_centered_binary_template(gray64):
    bin_center = binarize_center_legacy(gray64)
    out = np.full((64, 64), 255, dtype=np.uint8)
    out[16:48, 16:48] = (1 - bin_center) * 255
    return out


def main():
    parser = argparse.ArgumentParser(description="Update work.jpg template from ROI crops")
    parser.add_argument("--image", help="Single image path to use")
    parser.add_argument("--images", nargs="+", help="Multiple image paths to use")
    parser.add_argument("--average", action="store_true", help="Average multiple ROIs when --images is provided")
    parser.add_argument("--out", default="pyton/Templates/work.jpg", help="Output template path")
    parser.add_argument("--roi-list", default=None, help="Override ROI list path (single ROI list for all images)")
    parser.add_argument("--use-bin", action="store_true", help="Use legacy binarized ROI center for the template")
    args = parser.parse_args()

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    image_list = []
    if args.image:
        image_list = [args.image]
    elif args.images:
        image_list = list(args.images)
    else:
        raise RuntimeError("Provide --image or --images")

    if len(image_list) > 1 and not args.average:
        raise RuntimeError("Multiple images provided; use --average to combine them")

    roi_grays = []

    for img_path in image_list:
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        roi_list = resolve_roi_list(img_path, project_dir, args.roi_list)
        roi_gray = extract_roi_gray(img_path, roi_list)
        if args.use_bin:
            roi_gray = build_centered_binary_template(roi_gray)
        roi_grays.append(roi_gray)

    if not roi_grays:
        raise RuntimeError("No ROI images collected")

    if len(roi_grays) == 1:
        avg_u8 = roi_grays[0]
    else:
        avg = np.mean(np.stack(roi_grays, axis=0), axis=0)
        avg_u8 = np.clip(avg, 0, 255).astype(np.uint8)

    out_path = os.path.join(project_dir, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(avg_u8, mode="L").convert("RGB").save(out_path)

    print(f"Wrote updated template: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
