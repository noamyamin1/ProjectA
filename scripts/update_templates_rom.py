import os
import numpy as np
from PIL import Image

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
TARGET_H = 64
TARGET_W = 64
BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATE_DIR = os.path.join(BASE_DIR, "pyton/Templates")

MAPPING_FILE = os.path.join(DATA_DIR, "template_mapping.txt")
OUTPUT_MEM   = os.path.join(DATA_DIR, "templates.mem")

# ------------------------------------------------------------
# 1. Processing Logic
# ------------------------------------------------------------
def rgb_to_gray_u8(img_rgb_u8):
    R = img_rgb_u8[:, :, 0].astype(np.uint32)
    G = img_rgb_u8[:, :, 1].astype(np.uint32)
    B = img_rgb_u8[:, :, 2].astype(np.uint32)
    gray = (77 * R + 150 * G + 29 * B) >> 8
    return np.clip(gray, 0, 255).astype(np.uint8)

def remove_dark_background(img_gray, bg_thresh=20):
    """
    Floods dark areas starting from corners to turn black background into white (255).
    This ensures that after binarization, the background becomes 0 (like the sky).
    """
    h, w = img_gray.shape
    out = img_gray.copy()
    q = [(0,0), (0, w-1), (h-1, 0), (h-1, w-1)]
    visited = set(q)
    
    while q:
        r, c = q.pop(0)
        if out[r, c] <= bg_thresh:
            out[r, c] = 255 
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return out

# ------------------------------------------------------------
# 2. Main Execution Logic
# ------------------------------------------------------------
def build_rom():
    templates_data = {}
    
    # Load and process existing templates based on mapping
    print(f"Reading mapping from {MAPPING_FILE}...")
    if not os.path.exists(MAPPING_FILE):
        print("Error: Mapping file not found!")
        return

    with open(MAPPING_FILE, 'r') as f:
        for line in f:
            if not line.strip().startswith("ID"): continue
            try:
                parts = line.split(":")
                id_val = int(parts[0].replace("ID", "").strip())
                fname = parts[1].split("-")[0].strip()
                
                img_path = os.path.join(TEMPLATE_DIR, fname)
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
                    
                    gray = rgb_to_gray_u8(np.array(img))
                    clean_gray = remove_dark_background(gray)
                    templates_data[id_val] = clean_gray
                    print(f"  [OK] Processed ID {id_val}: {fname}")
                else:
                    print(f"  [!] Missing file for ID {id_val}: {fname}")
            except Exception as e:
                print(f"  [!] Error parsing line '{line.strip()}': {e}")

    # Write to templates.mem (1024 lines: 32 templates * 32 rows)
    print(f"\nGenerating {OUTPUT_MEM}...")
    with open(OUTPUT_MEM, 'w') as f:
        for tid in range(32):
            if tid in templates_data:
                img_g = templates_data[tid]
                
                # Hardware alignment: mean of 64x64, then crop to 32x32
                mean_val = int(np.mean(img_g))
                thresh = max(0, mean_val - 15)
                
                # Crop center 32x32
                my, mx = int(TARGET_H * 0.25), int(TARGET_W * 0.25)
                crop = img_g[my:TARGET_H-my, mx:TARGET_W-mx]
                bin_crop = (crop < thresh).astype(np.uint8)
                
                f.write(f"// ID {tid}\n")
                for r in range(32):
                    row_val = 0
                    for c in range(32):
                        if bin_crop[r, c]:
                            row_val |= (1 << (31 - c))
                    f.write(f"{row_val:08x}\n")
            else:
                # Padding for unused Template slots to keep memory aligned
                f.write(f"// ID {tid} (Empty)\n")
                for _ in range(32):
                    f.write("00000000\n")

    print("\nDone! ROM is ready for VCS simulation.")

if __name__ == "__main__":
    build_rom()