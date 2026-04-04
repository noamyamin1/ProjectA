import sys
import os
import numpy as np
from PIL import Image, ImageDraw

def main(golden_path, rtl_path, out_dir):
    target_w, target_h = 1920, 1080
    expected_len = target_w * target_h
    
    try:
        with open(golden_path, 'r') as fg:
            golden_data = np.array([int(line.strip()) for line in fg if line.strip()], dtype=np.uint8)
        with open(rtl_path, 'r') as fr:
            rtl_data = np.array([int(line.strip()) for line in fr if line.strip()], dtype=np.uint8)
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)
        
    if len(golden_data) != expected_len or len(rtl_data) != expected_len:
        print(f"Error: Length mismatch. Expected {expected_len} (FHD).")
        sys.exit(1)
        
    golden_mask = golden_data.reshape((target_h, target_w))
    rtl_mask = rtl_data.reshape((target_h, target_w))
    
    diff = (golden_mask != rtl_mask)
    errors = int(np.sum(diff))
    status_str = "PASS" if errors == 0 else "FAIL"
    
    golden_img = Image.fromarray(golden_mask * 255).convert("RGB")
    rtl_img = Image.fromarray(rtl_mask * 255).convert("RGB")
    
    vis_diff = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    vis_diff[(golden_mask == 1) & (rtl_mask == 1)] = [255, 255, 255]
    vis_diff[(golden_mask == 0) & (rtl_mask == 1)] = [255, 0, 0]
    vis_diff[(golden_mask == 1) & (rtl_mask == 0)] = [0, 0, 255]
    diff_img = Image.fromarray(vis_diff)
    
    scale = 0.5
    sw, sh = int(target_w * scale), int(target_h * scale)
    golden_img = golden_img.resize((sw, sh), Image.NEAREST)
    rtl_img = rtl_img.resize((sw, sh), Image.NEAREST)
    diff_img = diff_img.resize((sw, sh), Image.NEAREST)
    
    header_h = 100
    comp_w = sw * 3
    comp_h = sh + header_h
    comp_img = Image.new('RGB', (comp_w, comp_h), color=(30, 30, 30))
    
    draw = ImageDraw.Draw(comp_img)
    stats_text = f"RED MASK VERIFICATION | Res: {target_w}x{target_h} | Total Pixels: {expected_len} | Mismatches: {errors} | Status: {status_str}"
    
    draw.text((20, 20), stats_text, fill=(255, 255, 255))
    draw.text((20, 70), "GOLDEN MODEL (SW)", fill=(200, 200, 200))
    draw.text((sw + 20, 70), "RTL OUTPUT (HW)", fill=(200, 200, 200))
    draw.text((sw*2 + 20, 70), "DIFFERENCE (White=Match, Red=HW Extra, Blue=SW Extra)", fill=(200, 200, 200))
    
    comp_img.paste(golden_img, (0, header_h))
    comp_img.paste(rtl_img, (sw, header_h))
    comp_img.paste(diff_img, (sw*2, header_h))
    
    os.makedirs(out_dir, exist_ok=True)
    vis_path = os.path.join(out_dir, "red_mask_comparison.png")
    comp_img.save(vis_path)
    
    stats_path = os.path.join(out_dir, "red_mask_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"STAGE: RED_MASK\n")
        f.write(f"RESOLUTION: {target_w}x{target_h}\n")
        f.write(f"TOTAL_PIXELS: {expected_len}\n")
        f.write(f"MISMATCHES: {errors}\n")
        f.write(f"STATUS: {status_str}\n")
        
    print(f"Comparison image saved to: {vis_path}")
    print(f"Statistics log saved to: {stats_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python verify_red_mask.py <golden.txt> <rtl.txt> <out_results_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])