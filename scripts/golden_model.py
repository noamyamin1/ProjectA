import os
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

# Configuration parameters
INPUT_IMAGE_PATH = "/users/epnyrk/Project/design/work/ProjectA/pyton/pics_to_test/slippery_road_redcar.jpg"
DATA_DIR = "data"

# File paths
IMAGE_IN_HEX   = os.path.join(DATA_DIR, "image_in2.hex")
MASK_OUT_REF   = os.path.join(DATA_DIR, "mask_out2.txt")
MORPH_OUT_REF  = os.path.join(DATA_DIR, "morph_out2.txt")

# Fixed FHD resolution
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# Red Mask Algorithm Parameters
MIN_RED_VAL = 15
MARGIN_SHIFT = 3

def generate_golden_reference(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Input file '{img_path}' not found.")
        return

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")

    try:
        # Load, convert, resize
        img = Image.open(img_path).convert('RGB')
        
        if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
            print(f"Warning: Image size is {img.size}. Resizing to FHD {TARGET_WIDTH}x{TARGET_HEIGHT}.")
            try:
                resample_method = Image.Resampling.LANCZOS
            except AttributeError:
                resample_method = Image.ANTIALIAS
                
            img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), resample_method)
        
        # Convert to numpy array
        img_np = np.array(img)
        
        # Split channels
        r = img_np[:,:,0]
        g = img_np[:,:,1]
        b = img_np[:,:,2]
        
        # Calculate dynamic margin
        margin = r >> MARGIN_SHIFT
        
        # ==========================================
        # 1. Generate image_in.hex
        # ==========================================
        print(f"Generating: {IMAGE_IN_HEX}...")
        with open(IMAGE_IN_HEX, 'w') as f:
            for y in range(TARGET_HEIGHT):
                for x in range(TARGET_WIDTH):
                    hex_string = f"00{r[y,x]:02x}{g[y,x]:02x}{b[y,x]:02x}\n"
                    f.write(hex_string)
        
        # ==========================================
        # 2. Implement Red Mask Algorithm
        # ==========================================
        print(f"Computing Red Mask...")
        cond1_dom_r_vs_g    = (r.astype(np.int16) > (g.astype(np.int16) + margin))
        cond2_dom_r_vs_b    = (r.astype(np.int16) > (b.astype(np.int16) + margin))
        cond3_orange_killer = (g.astype(np.int16) < (b.astype(np.int16) + margin))
        cond4_min_red       = (r > MIN_RED_VAL)
        
        red_mask_np = np.logical_and.reduce((cond1_dom_r_vs_g, cond2_dom_r_vs_b, cond3_orange_killer, cond4_min_red))
        
        # Generate mask_out.txt
        print(f"Generating: {MASK_OUT_REF}...")
        with open(MASK_OUT_REF, 'w') as f:
            for y in range(TARGET_HEIGHT):
                for x in range(TARGET_WIDTH):
                    bit = '1' if red_mask_np[y,x] else '0'
                    f.write(bit + '\n')
                    
        # ==========================================
        # 3. Implement Morphology
        # ==========================================
        print(f"Computing Morphology filter...")
        struct_3x3 = np.ones((3,3), dtype=bool)
        
        dilated_mask_np = binary_dilation(red_mask_np, structure=struct_3x3)
        final_morph_np = binary_erosion(dilated_mask_np, structure=struct_3x3)
        
        # Generate morph_out.txt
        print(f"Generating: {MORPH_OUT_REF}...")
        with open(MORPH_OUT_REF, 'w') as f:
            for y in range(TARGET_HEIGHT):
                for x in range(TARGET_WIDTH):
                    bit = '1' if final_morph_np[y,x] else '0'
                    f.write(bit + '\n')

        print("Success! All reference files ready.")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    generate_golden_reference(INPUT_IMAGE_PATH)