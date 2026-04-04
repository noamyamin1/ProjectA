import os
import sys


def parse_visual_inspection(path):
    pass_stems = []
    fail_stems = []
    skip_stems = []
    if not os.path.exists(path):
        return pass_stems, fail_stems, skip_stems
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
            if "pass" in verdict_l and "fail" not in verdict_l:
                pass_stems.append(stem)
            elif "fail" in verdict_l:
                fail_stems.append(stem)
            else:
                skip_stems.append(stem)
    return pass_stems, fail_stems, skip_stems


def parse_geom_filtered(path):
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
                xmin = int(toks[1])
                xmax = int(toks[2])
                ymin = int(toks[3])
                ymax = int(toks[4])
                area = int(toks[5])
                perimeter = int(toks[6])
            except ValueError:
                continue
            rows.append((label, xmin, xmax, ymin, ymax, area, perimeter))
    return rows


def percentile(values, pct):
    if not values:
        return None
    vals = sorted(values)
    if pct <= 0:
        return vals[0]
    if pct >= 100:
        return vals[-1]
    k = (len(vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def stats_summary(values):
    if not values:
        return {}
    return {
        "min": min(values),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "max": max(values),
    }


def fmt_summary(name, stats):
    if not stats:
        return f"{name}: n/a"
    return (
        f"{name}: min={stats['min']:.3f} p10={stats['p10']:.3f} p25={stats['p25']:.3f} "
        f"p50={stats['p50']:.3f} p75={stats['p75']:.3f} p90={stats['p90']:.3f} max={stats['max']:.3f}"
    )


def main(project_dir):
    inspection_path = os.path.join(project_dir, "results", "geo_filter_visual_inspection.txt")
    by_image_dir = os.path.join(project_dir, "results", "by_image")

    pass_stems, fail_stems, skip_stems = parse_visual_inspection(inspection_path)

    widths = []
    heights = []
    areas = []
    solids = []
    missing = []

    for stem in pass_stems:
        geom_path = os.path.join(by_image_dir, stem, "geom_filtered_golden.txt")
        if not os.path.exists(geom_path):
            missing.append(stem)
            continue
        rows = parse_geom_filtered(geom_path)
        for _label, xmin, xmax, ymin, ymax, area, _perim in rows:
            w = (xmax - xmin + 1)
            h = (ymax - ymin + 1)
            bbox_area = w * h
            solidity = (float(area) / float(bbox_area)) if bbox_area > 0 else 0.0
            widths.append(float(w))
            heights.append(float(h))
            areas.append(float(area))
            solids.append(solidity)

    w_stats = stats_summary(widths)
    h_stats = stats_summary(heights)
    a_stats = stats_summary(areas)
    s_stats = stats_summary(solids)

    print("Geo filter pass-set stats")
    print("==========================")
    print(f"Pass stems: {len(pass_stems)} | Missing geom_filtered_golden: {len(missing)}")
    if missing:
        print("Missing stems: " + ", ".join(missing))
    print("")
    print(fmt_summary("width", w_stats))
    print(fmt_summary("height", h_stats))
    print(fmt_summary("area", a_stats))
    print(fmt_summary("solidity", s_stats))
    print("")

    if not widths or not heights or not areas or not solids:
        print("No pass-set data to propose thresholds.")
        return

    min_w = int(min(widths))
    min_h = int(min(heights))
    min_area = int(min(areas))
    min_solidity = min(solids)

    print("Proposed thresholds (won't harm current pass set)")
    print("-------------------------------------------------")
    print(f"min_w >= {min_w}")
    print(f"min_h >= {min_h}")
    print(f"min_area >= {min_area}")
    print(f"min_solidity >= {min_solidity:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python geo_filter_stats.py [project_dir]")
        sys.exit(1)
    proj = sys.argv[1] if len(sys.argv) == 2 else os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    main(proj)
