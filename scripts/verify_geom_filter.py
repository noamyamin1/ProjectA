import sys
import os
import numpy as np
from PIL import Image, ImageDraw


def load_boxes(path):
    boxes = {}
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 5:
                continue
            lbl = int(toks[0])
            boxes[lbl] = {
                "xmin": int(toks[1]),
                "xmax": int(toks[2]),
                "ymin": int(toks[3]),
                "ymax": int(toks[4]),
            }
    return boxes


def choose_reference_image(image_dir, explicit_path=None):
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        return None
    preferred = os.path.join(image_dir, "6.png")
    if os.path.exists(preferred):
        return preferred
    return None


def draw_boxes(base_img, boxes, color):
    img = base_img.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    for lbl in sorted(boxes.keys()):
        b = boxes[lbl]
        d.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=color, width=3)
        d.text((b["xmin"], max(0, b["ymin"] - 12)), str(lbl), fill=color)
    return img


def draw_combined_boxes(base_img, golden_boxes, rtl_boxes):
    img = base_img.copy().convert("RGB")
    d = ImageDraw.Draw(img)

    for lbl in sorted(golden_boxes.keys()):
        b = golden_boxes[lbl]
        d.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=(0, 255, 0), width=3)
        d.text((b["xmin"], max(0, b["ymin"] - 12)), f"G:{lbl}", fill=(0, 255, 0))

    for lbl in sorted(rtl_boxes.keys()):
        b = rtl_boxes[lbl]
        d.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=(255, 0, 0), width=2)
        d.text((b["xmin"], min(img.height - 12, b["ymax"] + 2)), f"R:{lbl}", fill=(255, 0, 0))

    return img


def main(golden_path, rtl_path, out_dir, image_dir, image_path=None):
    golden = load_boxes(golden_path)
    rtl = load_boxes(rtl_path)

    labels = sorted(set(golden.keys()) | set(rtl.keys()))

    missing_in_rtl = []
    extra_in_rtl = []
    bbox_mismatches = []

    for lbl in labels:
        g = golden.get(lbl)
        r = rtl.get(lbl)
        if g is None:
            extra_in_rtl.append(lbl)
            continue
        if r is None:
            missing_in_rtl.append(lbl)
            continue

        for k in ["xmin", "xmax", "ymin", "ymax"]:
            if g[k] != r[k]:
                bbox_mismatches.append((lbl, k, g[k], r[k]))

    status = "PASS" if (not missing_in_rtl and not extra_in_rtl and not bbox_mismatches) else "FAIL"

    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(out_dir, "geom_filter_verify.txt")
    with open(report_path, "w") as f:
        f.write("STAGE: GEOMETRY_FILTER\n")
        f.write(f"GOLDEN_BOXES: {len(golden)}\n")
        f.write(f"RTL_BOXES: {len(rtl)}\n")
        f.write(f"MISSING_IN_RTL: {len(missing_in_rtl)}\n")
        f.write(f"EXTRA_IN_RTL: {len(extra_in_rtl)}\n")
        f.write(f"BBOX_MISMATCHES: {len(bbox_mismatches)}\n")
        f.write(f"STATUS: {status}\n")

        if missing_in_rtl:
            f.write("\nMissing labels in RTL:\n")
            for lbl in missing_in_rtl[:200]:
                f.write(f"{lbl}\n")

        if extra_in_rtl:
            f.write("\nExtra labels in RTL:\n")
            for lbl in extra_in_rtl[:200]:
                f.write(f"{lbl}\n")

        if bbox_mismatches:
            f.write("\nBBox mismatches (label field golden rtl):\n")
            for lbl, k, gv, rv in bbox_mismatches[:500]:
                f.write(f"{lbl} {k} {gv} {rv}\n")

    # Visualization
    ref_img_path = choose_reference_image(image_dir, image_path)
    if ref_img_path is not None:
        img = Image.open(ref_img_path).convert("RGB")
        w, h = img.size

        # Keep overlay coordinates meaningful: resize to FHD if needed.
        if (w, h) != (1920, 1080):
            img = img.resize((1920, 1080), Image.BILINEAR)

        golden_img = draw_boxes(img, golden, (0, 255, 0))
        rtl_img = draw_boxes(img, rtl, (255, 0, 0))

        header_h = 80
        sw, sh = 960, 540
        a = golden_img.resize((sw, sh), Image.BILINEAR)
        b = rtl_img.resize((sw, sh), Image.BILINEAR)

        comp = Image.new("RGB", (sw * 2, sh + header_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(comp)
        draw.text((20, 20), f"GEOMETRY FILTER | Golden: {len(golden)} | RTL: {len(rtl)} | Status: {status}", fill=(255, 255, 255))
        draw.text((20, 55), "Golden Boxes (Green)", fill=(200, 200, 200))
        draw.text((sw + 20, 55), "RTL Boxes (Red)", fill=(200, 200, 200))
        comp.paste(a, (0, header_h))
        comp.paste(b, (sw, header_h))

        vis_path = os.path.join(out_dir, "geom_filter_comparison.png")
        comp.save(vis_path)
        print(f"Visualization: {vis_path}")

        combined = draw_combined_boxes(img, golden, rtl)
        overlay_header_h = 80
        ow, oh = 1280, 720
        combined_small = combined.resize((ow, oh), Image.BILINEAR)
        overlay = Image.new("RGB", (ow, oh + overlay_header_h), color=(30, 30, 30))
        od = ImageDraw.Draw(overlay)
        od.text((20, 20), f"Selected BBoxes Overlay (6.png) | Green=Golden | Red=RTL | Status={status}", fill=(255, 255, 255))
        overlay.paste(combined_small, (0, overlay_header_h))
        overlay_path = os.path.join(out_dir, "geom_filter_overlay.png")
        overlay.save(overlay_path)
        print(f"Visualization: {overlay_path}")
    else:
        if image_path:
            print(f"Warning: image not found: {image_path}; skipped visualization")
        else:
            print(f"Warning: 6.png not found in {image_dir}; skipped visualization")

    print("----------------------------------------")
    print("GEOMETRY FILTER VERIFICATION")
    print("----------------------------------------")
    print(f"Golden boxes: {len(golden)}")
    print(f"RTL boxes: {len(rtl)}")
    print(f"Missing in RTL: {len(missing_in_rtl)}")
    print(f"Extra in RTL: {len(extra_in_rtl)}")
    print(f"BBox mismatches: {len(bbox_mismatches)}")
    print(f"Overall Status: {status}")
    print(f"Report: {report_path}")

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        print("Usage: python verify_geom_filter.py <geom_bboxes_golden.txt> <actual_geom_boxes.txt> <out_results_dir> <image_dir> [image_path]")
        sys.exit(1)
    image_path = sys.argv[5] if len(sys.argv) == 6 else None
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], image_path)
