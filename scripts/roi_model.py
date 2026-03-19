import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------
# Configuration & Paths
# ------------------------------------------------------------
TARGET_H = 64
TARGET_W = 64

BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATE_DIR = os.path.join(BASE_DIR, "pyton/Templates") 
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MAPPING_FILE = os.path.join(DATA_DIR, "template_mapping.txt")
RTL_ROI_HEX  = os.path.join(DATA_DIR, "rtl_fetched_gray.hex")

# ------------------------------------------------------------
# 1. Load Data from RTL
# ------------------------------------------------------------
def load_rtl_gray_roi(filepath=RTL_ROI_HEX, h=TARGET_H, w=TARGET_W):
    img = np.zeros((h, w), dtype=np.uint8)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            for i, val in enumerate(lines):
                if i < h * w:
                    img[i // w, i % w] = int(val, 16)
        print(f"Loaded RTL ROI from {filepath}")
    else:
        print(f"Warning: {filepath} not found. Using a blank ROI for testing.")
    return img

# ------------------------------------------------------------
# 2. Template Loading (Mapped directly to HW IDs)
# ------------------------------------------------------------
def rgb_to_gray_u8(img_rgb_u8):
    R = img_rgb_u8[:, :, 0].astype(np.uint32)
    G = img_rgb_u8[:, :, 1].astype(np.uint32)
    B = img_rgb_u8[:, :, 2].astype(np.uint32)
    
    gray = (77 * R + 150 * G + 29 * B) >> 8
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

def load_mapped_templates(mapping_file, template_dir, out_h=TARGET_H, out_w=TARGET_W):
    templates = {}
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file '{mapping_file}' not found.")
        
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("ID"): continue
            
            parts = line.split(":")
            if len(parts) < 2: continue
            
            id_part = parts[0].strip()
            id_val = int(id_part.replace("ID", "").strip())
            
            file_desc_part = parts[1].strip()
            fname = file_desc_part.split("-")[0].strip()
            desc = file_desc_part.split("-")[1].strip() if "-" in file_desc_part else "Unknown"
            
            img_path = os.path.join(template_dir, fname)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                
                try:
                    resample_method = Image.Resampling.LANCZOS
                except AttributeError:
                    resample_method = Image.ANTIALIAS
                    
                img = img.resize((out_w, out_h), resample_method)
                img_np = np.array(img)
                
                # המרה לשחור-לבן
                gray = rgb_to_gray_u8(img_np)
                
                # >>> התיקון שלנו: הסרת הרקע השחור מהתבנית <<<
                gray_cleaned = remove_dark_background(gray)
                
                label = f"{id_val:02d}_{desc}"
                templates[label] = gray_cleaned
            else:
                print(f"Warning: Template image {fname} not found in {template_dir}")
                
    return templates

def remove_dark_background(img_gray, bg_thresh=20):
    """
    מבצע Flood-Fill (מילוי שטחים) מ-4 הפינות של התמונה.
    אם הפינה חשוכה (רקע שחור של תבנית), הוא צובע אותה בלבן (255)
    כדי שהבינאריזציה תהפוך אותה ל-0 כמו השמיים במציאות.
    """
    h, w = img_gray.shape
    out = img_gray.copy()
    
    # תור של 4 הפינות להתחלת הסריקה
    q = [(0,0), (0, w-1), (h-1, 0), (h-1, w-1)]
    visited = set(q)
    
    while q:
        r, c = q.pop(0)
        # אם הפיקסל חשוך (רקע), הפוך אותו לבהיר והמשך לשכנים
        if out[r, c] <= bg_thresh:
            out[r, c] = 255 
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return out

# ------------------------------------------------------------
# 3. Core Matching Logic (Updated to match HW Mismatch Count)
# ------------------------------------------------------------
def get_binary_crop_for_vis(img_u8, margin_ratio=0.25):
    H, W = img_u8.shape
    my = int(H * margin_ratio)
    mx = int(W * margin_ratio)
    
    crop = img_u8[my:H-my, mx:W-mx]
    if crop.size == 0:
        return np.zeros_like(img_u8)
        
    mean_val = int(np.mean(crop))
    bin_crop = (crop < (mean_val - 15)).astype(np.uint8) * 255
    return bin_crop

def calculate_hw_mismatches(a_u8, b_u8, margin_ratio=0.25):
    H, W = a_u8.shape
    my = int(H * margin_ratio)
    mx = int(W * margin_ratio)
    
    a_crop = a_u8[my:H-my, mx:W-mx]
    b_crop = b_u8[my:H-my, mx:W-mx]
    
    a_mean = int(np.mean(a_crop))
    b_mean = int(np.mean(b_crop))
    
    a_bin = (a_crop < (a_mean - 15)).astype(np.uint8) 
    b_bin = (b_crop < (b_mean - 15)).astype(np.uint8) 
    
    best_mismatches = float('inf')
    cH, cW = a_bin.shape
    
    # Simulating the exact 9x9 hardware shifting
    for dy in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
        for dx in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            y_start_a = max(0, dy)
            y_end_a = min(cH, cH + dy)
            x_start_a = max(0, dx)
            x_end_a = min(cW, cW + dx)
            
            y_start_b = max(0, -dy)
            y_end_b = min(cH, cH - dy)
            x_start_b = max(0, -dx)
            x_end_b = min(cW, cW - dx)
            
            a_shifted = a_bin[y_start_a:y_end_a, x_start_a:x_end_a]
            b_shifted = b_bin[y_start_b:y_end_b, x_start_b:x_end_b]
            
            mismatches = np.sum(np.bitwise_xor(a_shifted, b_shifted))
            
            if mismatches < best_mismatches:
                best_mismatches = mismatches
                
    # Return absolute mismatch count just like RTL reports
    return int(best_mismatches)

# ------------------------------------------------------------
# 4. Visualization & Execution
# ------------------------------------------------------------
def run_and_visualize(roi_g, templates_gray_u8, top_k=3):
    scores = []
    
    for lab, tmpl_img in templates_gray_u8.items():
        mismatch_count = calculate_hw_mismatches(roi_g, tmpl_img, margin_ratio=0.25)
        scores.append((lab, mismatch_count))

    scores.sort(key=lambda t: t[1])
    topk = scores[:min(top_k, len(scores))]
    
    if not topk:
        print("No matches calculated.")
        return

    best_lab, best_sc = topk[0]
    second_sc = topk[1][1] if len(topk) >= 2 else None
    margin = (second_sc - best_sc) if second_sc is not None else None

    print(f"\nRTL ROI Match Results:")
    print(f"Best Match: '{best_lab}' with {best_sc} Mismatches")
    if margin is not None:
        print(f"2nd Best: {second_sc} Mismatches (Margin: {margin})")

    roi_bin = get_binary_crop_for_vis(roi_g)
    
    fig, axes = plt.subplots(3, top_k + 1, figsize=(3.0 * (top_k + 1), 9.0))

    # Row 0: Original Grayscale
    axes[0, 0].imshow(roi_g, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("RTL ROI Original")
    axes[0, 0].axis("off")

    for j, (lab, sc) in enumerate(topk, start=1):
        tmpl_g = templates_gray_u8[lab]
        axes[0, j].imshow(tmpl_g, cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(f"Tmpl: {lab}\nMismatches={sc}")
        axes[0, j].axis("off")

    # Row 1: Binarized
    axes[1, 0].imshow(roi_bin, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("ROI Binary\n(HW View)")
    axes[1, 0].axis("off")

    for j, (lab, sc) in enumerate(topk, start=1):
        tmpl_bin = get_binary_crop_for_vis(templates_gray_u8[lab])
        axes[1, j].imshow(tmpl_bin, cmap="gray", vmin=0, vmax=255)
        axes[1, j].set_title("Tmpl Binary")
        axes[1, j].axis("off")

    # Row 2: XOR Mismatch Map (Unshifted)
    axes[2, 0].imshow(roi_bin, cmap="gray", vmin=0, vmax=255)
    axes[2, 0].set_title("ROI Binary")
    axes[2, 0].axis("off")

    for j, (lab, sc) in enumerate(topk, start=1):
        tmpl_bin = get_binary_crop_for_vis(templates_gray_u8[lab])
        min_h = min(roi_bin.shape[0], tmpl_bin.shape[0])
        min_w = min(roi_bin.shape[1], tmpl_bin.shape[1])
        diff = np.bitwise_xor(roi_bin[:min_h, :min_w], tmpl_bin[:min_h, :min_w])
        axes[2, j].imshow(diff, cmap="magma", vmin=0, vmax=255)
        axes[2, j].set_title("XOR Mismatches")
        axes[2, j].axis("off")

    plt.tight_layout()
    
    # Save image instead of blocking with plt.show() on Linux servers
    out_img = os.path.join(RESULTS_DIR, "golden_match_results.png")
    plt.savefig(out_img)
    print(f"\nVisualization saved to {out_img}")

if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")  
    templates_dict = load_mapped_templates(MAPPING_FILE, TEMPLATE_DIR)
    rtl_roi_img = load_rtl_gray_roi()
    run_and_visualize(rtl_roi_img, templates_dict, top_k=3)
