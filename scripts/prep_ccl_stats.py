import sys
import os
import numpy as np


def read_labels(path):
    with open(path, "r") as f:
        vals = [int(line.strip()) for line in f if line.strip()]
    return np.array(vals, dtype=np.uint16)


def compute_component_stats(labels_2d, label_id):
    ys, xs = np.where(labels_2d == label_id)
    area = int(len(xs))
    if area == 0:
        return None

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())

    perimeter = 0
    h, w = labels_2d.shape
    for y, x in zip(ys, xs):
        edge = False
        if y == 0 or labels_2d[y - 1, x] != label_id:
            edge = True
        elif y == h - 1 or labels_2d[y + 1, x] != label_id:
            edge = True
        elif x == 0 or labels_2d[y, x - 1] != label_id:
            edge = True
        elif x == w - 1 or labels_2d[y, x + 1] != label_id:
            edge = True

        if edge:
            perimeter += 1

    return (int(label_id), area, perimeter, xmin, xmax, ymin, ymax)


def main(ccl_pass2_path, out_dir):
    width = 1920
    height = 1080
    total = width * height

    labels = read_labels(ccl_pass2_path)
    if len(labels) != total:
        print(f"Error: expected {total} labels, got {len(labels)}")
        sys.exit(1)

    labels_2d = labels.reshape((height, width))

    y_offset = 0

    uniq = np.unique(labels)
    uniq = uniq[uniq != 0]

    stats = []
    for lbl in uniq:
        row = compute_component_stats(labels_2d, int(lbl))
        if row is not None:
            label, area, perimeter, xmin, xmax, ymin, ymax = row
            ymin_adj = max(0, ymin - y_offset)
            ymax_adj = max(0, ymax - y_offset)
            stats.append((label, area, perimeter, xmin, xmax, ymin_adj, ymax_adj))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ccl_stats_golden.txt")
    with open(out_path, "w") as f:
        f.write("# label area perimeter xmin xmax ymin ymax\n")
        for row in stats:
            f.write("{} {} {} {} {} {} {}\n".format(*row))

    print(f"Generated {len(stats)} component stats -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prep_ccl_stats.py <actual_ccl_pass2.txt> <out_data_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
