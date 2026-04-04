import os
import sys
import numpy as np
from PIL import Image


def label_to_color(label):
    if label == 0:
        return (0, 0, 0)
    v = (label * 2654435761) & 0xFFFFFFFF
    r = (v >> 16) & 0xFF
    g = (v >> 8) & 0xFF
    b = v & 0xFF
    if r < 40 and g < 40 and b < 40:
        r = (r + 80) & 0xFF
        g = (g + 80) & 0xFF
        b = (b + 80) & 0xFF
    return (r, g, b)


def main(labels_path, out_path):
    width = 1920
    height = 1080
    expected_len = width * height

    with open(labels_path, "r") as f:
        data = [int(line.strip()) for line in f if line.strip()]

    if len(data) != expected_len:
        print(f"Error: expected {expected_len} labels, got {len(data)}")
        sys.exit(1)

    labels = np.array(data, dtype=np.uint16).reshape((height, width))
    unique = np.unique(labels)

    colors = np.zeros((len(unique), 3), dtype=np.uint8)
    for i, lbl in enumerate(unique):
        colors[i] = label_to_color(int(lbl))

    idx = np.searchsorted(unique, labels)
    rgb = colors[idx]

    img = Image.fromarray(rgb, mode="RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    print(f"CCL resolved visualization saved to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python visualize_ccl_resolved.py <actual_ccl_pass2.txt> <out_png>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
