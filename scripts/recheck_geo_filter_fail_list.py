import os
import sys


def parse_visual_inspection(path):
    fails = []
    if not os.path.exists(path):
        return fails
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if " - " not in s:
                continue
            stem, verdict = s.split(" - ", 1)
            stem = stem.strip()
            verdict_l = verdict.lower()
            if "fail" in verdict_l:
                fails.append(stem)
    return fails


def load_stats(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 7:
                continue
            try:
                label = int(toks[0])
                area = int(toks[1])
                perimeter = int(toks[2])
                xmin = int(toks[3])
                xmax = int(toks[4])
                ymin = int(toks[5])
                ymax = int(toks[6])
            except ValueError:
                continue
            rows.append((label, area, perimeter, xmin, xmax, ymin, ymax))
    return rows


def passes_geometry_filter(area, perimeter, xmin, xmax, ymin, ymax,
                           min_area, max_area, min_w, min_h, min_pix_area, min_solidity,
                           circ_min_num=12566, circ_min_den=100,
                           fill_max_num=60, fill_max_den=100):
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return False
    spatial_area = width * height
    pass_min_area = (spatial_area >= min_area) and (spatial_area <= max_area)
    pass_min_dims = (width >= min_w) and (height >= min_h)
    pass_min_pix_area = (area >= min_pix_area)
    pass_aspect = ((width * 4) >= (height * 3)) and ((width * 3) <= (height * 4))
    perim_sq = perimeter * perimeter
    pass_circularity = (circ_min_num * area) >= (circ_min_den * perim_sq)
    pass_fill_ratio = (area * fill_max_den) <= (spatial_area * fill_max_num)
    pass_fill_min = (area / float(spatial_area)) >= min_solidity if spatial_area > 0 else False
    return pass_min_area and pass_min_dims and pass_min_pix_area and pass_aspect and pass_circularity and pass_fill_ratio and pass_fill_min


def choose_label(rows, thresholds):
    min_area, max_area, min_w, min_h, min_pix_area, min_solidity = thresholds
    passed = []
    for label, area, perim, xmin, xmax, ymin, ymax in rows:
        if passes_geometry_filter(area, perim, xmin, xmax, ymin, ymax,
                                  min_area, max_area, min_w, min_h, min_pix_area, min_solidity):
            passed.append((label, area))
    if not passed:
        return None
    return max(passed, key=lambda r: r[1])[0]


def load_selected_label(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 1:
                continue
            try:
                return int(toks[0])
            except ValueError:
                return None
    return None


def main(project_dir):
    inspection_path = os.path.join(project_dir, "results", "geo_filter_visual_inspection.txt")
    by_image_dir = os.path.join(project_dir, "results", "by_image")

    fails = parse_visual_inspection(inspection_path)

    hard = (300, 100000, 34, 32, 313, 0.218)
    soft = (300, 100000, 35, 33, 350, 0.25)

    print("Fail list recheck (label selection)")
    print("=================================")
    print("Format: stem | old_label -> hard_label -> soft_label")

    for stem in fails:
        stats_path = os.path.join(by_image_dir, stem, "ccl_stats_golden.txt")
        rows = load_stats(stats_path)
        if not rows:
            print(f"{stem} | no stats")
            continue
        old_path = os.path.join(by_image_dir, stem, "geom_filtered_golden.txt")
        old_label = load_selected_label(old_path)
        hard_label = choose_label(rows, hard)
        soft_label = choose_label(rows, soft)
        print(f"{stem} | {old_label} -> {hard_label} -> {soft_label}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python recheck_geo_filter_fail_list.py [project_dir]")
        sys.exit(1)
    proj = sys.argv[1] if len(sys.argv) == 2 else os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    main(proj)
