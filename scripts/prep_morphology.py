import sys
import os
import numpy as np

def simulate_morphology_rtl(mask_stream, width, height):
    def run_stage(in_stream, is_dilation):
        out_stream = np.zeros_like(in_stream)
        line_buf_0 = np.zeros(width, dtype=np.uint8)
        line_buf_1 = np.zeros(width, dtype=np.uint8)
        
        w = np.zeros((3,3), dtype=np.uint8)
        wr_ptr = 0
        line_count = 0
        
        for i in range(len(in_stream)):
            s_data = in_stream[i]
            
            rdata_0 = line_buf_0[wr_ptr] if line_count >= 1 else 0
            rdata_1 = line_buf_1[wr_ptr] if line_count == 2 else 0
            
            line_buf_1[wr_ptr] = rdata_0
            line_buf_0[wr_ptr] = s_data
            
            w[0][2] = w[0][1]; w[0][1] = w[0][0]; w[0][0] = s_data
            w[1][2] = w[1][1]; w[1][1] = w[1][0]; w[1][0] = rdata_0
            w[2][2] = w[2][1]; w[2][1] = w[2][0]; w[2][0] = rdata_1
            
            if is_dilation:
                out_stream[i] = 1 if w.any() else 0
            else:
                out_stream[i] = 1 if w.all() else 0
                
            wr_ptr += 1
            if wr_ptr == width:
                wr_ptr = 0
                if line_count < 2:
                    line_count += 1
        return out_stream

    print("  -> Running HW-accurate Dilation...")
    dilated = run_stage(mask_stream, is_dilation=True)
    print("  -> Running HW-accurate Erosion...")
    eroded = run_stage(dilated, is_dilation=False)
    return eroded

def main(mask_in_path, out_dir):
    target_w, target_h = 1920, 1080
    expected_len = target_w * target_h
    
    try:
        with open(mask_in_path, 'r') as f:
            mask_stream = np.array([int(line.strip()) for line in f if line.strip()], dtype=np.uint8)
    except Exception as e:
        print(f"Error loading input mask: {e}")
        sys.exit(1)
        
    if len(mask_stream) != expected_len:
        print(f"Error: Mask length {len(mask_stream)} does not match FHD ({expected_len}).")
        sys.exit(1)
        
    golden_out = os.path.join(out_dir, "morph_out.txt")
    
    morph_stream = simulate_morphology_rtl(mask_stream, target_w, target_h)
    
    with open(golden_out, 'w') as f_gold:
        for val in morph_stream:
            f_gold.write(f"{val}\n")
            
    print(f"Morphology pre-processing complete.")
    print(f"Golden Model saved to: {golden_out}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prep_morphology.py <input_mask.txt> <out_data_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])