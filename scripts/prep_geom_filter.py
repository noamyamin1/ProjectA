import sys
import os


def load_stats(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 7:
                continue
            label = int(toks[0])
            area = int(toks[1])
            perimeter = int(toks[2])
            xmin = int(toks[3])
            xmax = int(toks[4])
            ymin = int(toks[5])
            ymax = int(toks[6])
            rows.append((label, area, perimeter, xmin, xmax, ymin, ymax))
    return rows


def passes_geometry_filter(area, perimeter, xmin, xmax, ymin, ymax, min_area, max_area,
                           min_w=32, min_h=32, min_pix_area=313, min_solidity=0.217,
                           min_w_relax=31, min_h_relax=30, relax_solidity=0.40,
                           aspect_relax_num=22, aspect_relax_den=10,
                           circ_min_num=12566, circ_min_den=100,
                           fill_max_num=60, fill_max_den=100):
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    if width <= 0 or height <= 0:
        return False

    spatial_area = width * height
    pass_min_area = (spatial_area >= min_area) and (spatial_area <= max_area)
    pass_min_dims = (width >= min_w) and (height >= min_h)
    pass_min_dims_relax = (width >= min_w_relax) and (height >= min_h_relax) and (area / float(spatial_area)) >= relax_solidity
    pass_min_dims_final = pass_min_dims or pass_min_dims_relax
    pass_min_pix_area = (area >= min_pix_area)
    pass_aspect = ((width * 4) >= (height * 3)) and ((width * 3) <= (height * 4))
    pass_aspect_relax = (width * aspect_relax_den <= height * aspect_relax_num) and (height * aspect_relax_den <= width * aspect_relax_num)
    pass_aspect_final = pass_aspect or (pass_aspect_relax and pass_min_dims_relax)
    perim_sq = perimeter * perimeter
    pass_circularity = (circ_min_num * area) >= (circ_min_den * perim_sq)
    pass_fill_ratio = (area * fill_max_den) <= (spatial_area * fill_max_num)
    pass_fill_min = (area / float(spatial_area)) >= min_solidity if spatial_area > 0 else False

    return pass_min_area and pass_min_dims_final and pass_min_pix_area and pass_aspect_final and pass_circularity and pass_fill_ratio and pass_fill_min


def compute_metrics(area, perimeter, xmin, xmax, ymin, ymax):
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    spatial_area = width * height
    perim_sq = perimeter * perimeter
    return width, height, spatial_area, perim_sq


def main(stats_path, out_dir, min_area=300, max_area=100000,
         min_w=34, min_h=32, min_pix_area=313, min_solidity=0.217,
         min_w_relax=31, min_h_relax=30, relax_solidity=0.40,
         aspect_relax_num=22, aspect_relax_den=10,
         max_candidates=5):
    rows = load_stats(stats_path)

    debug_rows = []

    passed = []
    pass_count = 0
    circ_min_num = 12566
    circ_min_den = 100
    fill_max_num = 60
    fill_max_den = 100

    for label, area, perimeter, xmin, xmax, ymin, ymax in rows:
        width, height, spatial_area, perim_sq = compute_metrics(area, perimeter, xmin, xmax, ymin, ymax)

        pass_min_area = (spatial_area >= min_area) and (spatial_area <= max_area)
        pass_min_dims = (width >= min_w) and (height >= min_h)
        pass_min_dims_relax = (width >= min_w_relax) and (height >= min_h_relax) and (area / float(spatial_area)) >= relax_solidity
        pass_min_dims_final = pass_min_dims or pass_min_dims_relax
        pass_min_pix_area = (area >= min_pix_area)
        pass_aspect = ((width * 4) >= (height * 3)) and ((width * 3) <= (height * 4))
        pass_aspect_relax = (width * aspect_relax_den <= height * aspect_relax_num) and (height * aspect_relax_den <= width * aspect_relax_num)
        pass_aspect_final = pass_aspect or (pass_aspect_relax and pass_min_dims_relax)
        pass_circ = (circ_min_num * area) >= (circ_min_den * perim_sq)
        pass_fill = (area * fill_max_den) <= (spatial_area * fill_max_num)
        pass_fill_min = (area / float(spatial_area)) >= min_solidity if spatial_area > 0 else False

        debug_rows.append((
            label, area, perimeter, xmin, xmax, ymin, ymax,
            width, height, spatial_area,
            int(pass_min_area), int(pass_min_dims_final), int(pass_min_pix_area),
            int(pass_aspect_final), int(pass_circ), int(pass_fill), int(pass_fill_min)
        ))

        if pass_min_area and pass_min_dims_final and pass_min_pix_area and pass_aspect_final and pass_circ and pass_fill and pass_fill_min:
            pass_count += 1
            passed.append((label, xmin, xmax, ymin, ymax, area, perimeter, perim_sq, width, height))

    os.makedirs(out_dir, exist_ok=True)

    out_filtered = os.path.join(out_dir, "geom_filtered_golden.txt")
    out_bboxes = os.path.join(out_dir, "geom_bboxes_golden.txt")

    if pass_count > max_candidates:
        passed = []
    elif passed:
        def better(a, b):
            area_a, perim_sq_a, w_a, h_a = a[5], a[7], a[8], a[9]
            area_b, perim_sq_b, w_b, h_b = b[5], b[7], b[8], b[9]
            spatial_a = w_a * h_a
            spatial_b = w_b * h_b
            fill_lhs = area_a * spatial_b
            fill_rhs = area_b * spatial_a
            if fill_lhs != fill_rhs:
                return fill_lhs < fill_rhs
            lhs = area_a * perim_sq_b
            rhs = area_b * perim_sq_a
            if lhs != rhs:
                return lhs > rhs
            delta_a = abs(w_a - h_a)
            delta_b = abs(w_b - h_b)
            if delta_a != delta_b:
                return delta_a < delta_b
            return area_a > area_b

        best = passed[0]
        for cand in passed[1:]:
            if better(cand, best):
                best = cand
        passed = [best[:7]]

    with open(out_filtered, "w") as f:
        f.write("# label xmin xmax ymin ymax area perimeter\n")
        for row in passed:
            f.write("{} {} {} {} {} {} {}\n".format(*row))

    with open(out_bboxes, "w") as f:
        f.write("# label xmin xmax ymin ymax\n")
        for label, xmin, xmax, ymin, ymax, area, perimeter in passed:
            f.write(f"{label} {xmin} {xmax} {ymin} {ymax}\n")

    debug_path = os.path.join(out_dir, "geom_filter_debug.txt")
    with open(debug_path, "w") as f:
        f.write("# label area perimeter xmin xmax ymin ymax width height spatial_area pass_area pass_dims pass_pix_area pass_aspect pass_circ pass_fill pass_fill_min\n")
        # Sort by area descending for easier inspection
        for row in sorted(debug_rows, key=lambda r: r[1], reverse=True):
            f.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(*row))

    print(f"Geometry filter kept {len(passed)} components")
    print(f"Debug report: {debug_path}")
    print(f"Wrote {out_filtered}")
    print(f"Wrote {out_bboxes}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 13:
        print("Usage: python prep_geom_filter.py <ccl_stats_golden.txt> <out_data_dir> [min_area] [max_area] [min_w] [min_h] [min_pix_area] [min_solidity] [min_w_relax] [min_h_relax] [relax_solidity] [max_candidates]")
        sys.exit(1)

    min_area = int(sys.argv[3]) if len(sys.argv) >= 4 else 300
    max_area = int(sys.argv[4]) if len(sys.argv) >= 5 else 100000
    min_w = int(sys.argv[5]) if len(sys.argv) >= 6 else 34
    min_h = int(sys.argv[6]) if len(sys.argv) >= 7 else 32
    min_pix_area = int(sys.argv[7]) if len(sys.argv) >= 8 else 313
    min_solidity = float(sys.argv[8]) if len(sys.argv) >= 9 else 0.217
    min_w_relax = int(sys.argv[9]) if len(sys.argv) >= 10 else 31
    min_h_relax = int(sys.argv[10]) if len(sys.argv) >= 11 else 30
    relax_solidity = float(sys.argv[11]) if len(sys.argv) >= 12 else 0.40
    max_candidates = int(sys.argv[12]) if len(sys.argv) >= 13 else 5
    main(
        sys.argv[1], sys.argv[2],
        min_area, max_area,
        min_w, min_h, min_pix_area, min_solidity,
        min_w_relax, min_h_relax, relax_solidity,
        22, 10, max_candidates
    )
