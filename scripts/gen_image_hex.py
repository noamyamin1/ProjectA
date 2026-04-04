import argparse
import os
from PIL import Image
import numpy as np


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


def iter_images(inputs, list_file=None, input_dir=None):
    items = []
    if list_file:
        with open(list_file, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                items.append(s)
    if input_dir:
        for name in sorted(os.listdir(input_dir)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                items.append(os.path.join(input_dir, name))
    items.extend(inputs)
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Generate image_in.hex files for integration TB."
    )
    parser.add_argument("inputs", nargs="*", help="Image paths")
    parser.add_argument("--list-file", help="Text file with one image path per line")
    parser.add_argument("--dir", dest="input_dir", help="Directory with images")
    parser.add_argument(
        "--out-root",
        default="results/by_image",
        help="Output root for per-image folders",
    )
    parser.add_argument(
        "--project-root",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
        help="Project root (default: repo root)",
    )
    args = parser.parse_args()

    out_root = os.path.abspath(os.path.join(args.project_root, args.out_root))
    os.makedirs(out_root, exist_ok=True)

    images = iter_images(args.inputs, args.list_file, args.input_dir)
    if not images:
        raise SystemExit("No images provided. Use --dir or pass image paths.")

    for img_path in images:
        if not os.path.exists(img_path):
            print(f"WARNING: missing image: {img_path}")
            continue

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(out_root, stem)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "image_in.hex")

        img = Image.open(img_path).convert("RGB")
        img_np = resize_image_fhd(np.array(img, dtype=np.uint8))
        write_image_hex(img_np, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
