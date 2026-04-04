import os
import sys
import numpy as np
from PIL import Image

def red_mask_lowlevel_single(img_rgb_u8, min_red_val=15, margin_shift=3):
    img_int = img_rgb_u8.astype(np.int16)
    R = img_int[:, :, 0]
    G = img_int[:, :, 1]
    B = img_int[:, :, 2]
    
    margin = R // (2 ** margin_shift)
    
    cond_R_dom_G = (R - G) > margin
    cond_R_dom_B = (R - B) > margin
    cond_not_orange = G < (B + margin)
    cond_min_val = R > min_red_val
    
    mask = (cond_R_dom_G & cond_R_dom_B & cond_not_orange & cond_min_val).astype(np.uint8)
    return mask

def main(image_path, out_dir):
    target_w, target_h = 1920, 1080
    
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((target_w, target_h), Image.BILINEAR)
        img_np = np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
    hex_out = os.path.join(out_dir, "image_in.hex")
    golden_out = os.path.join(out_dir, "mask_out.txt")
    
    with open(hex_out, 'w') as f_hex:
        for y in range(target_h):
            for x in range(target_w):
                r, g, b = img_np[y, x]
                f_hex.write(f"{r:02x}{g:02x}{b:02x}\n")
                
    mask = red_mask_lowlevel_single(img_np)
    
    with open(golden_out, 'w') as f_gold:
        for y in range(target_h):
            for x in range(target_w):
                f_gold.write(f"{mask[y, x]}\n")
                
    print(f"Pre-processing complete. Forced Resolution: {target_w}x{target_h}")
    print(f"TB Input saved to: {hex_out}")
    print(f"Golden Model saved to: {golden_out}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prep_red_mask.py <input_image_path> <out_data_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])