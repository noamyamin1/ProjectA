import sys
import os
import numpy as np

def hw_accurate_ccl(mask_stream, width, height):
    total_pixels = width * height
    labels_pass1 = np.zeros(total_pixels, dtype=np.uint16)
    parent = np.zeros(65536, dtype=np.uint16)
    next_label = 1
    
    # PASS 1
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if mask_stream[idx] == 1:
                left = labels_pass1[idx - 1] if x > 0 else 0
                up = labels_pass1[idx - width] if y > 0 else 0
                
                if left == 0 and up == 0:
                    if next_label < 65535:
                        out_lbl = next_label
                        parent[next_label] = next_label
                        next_label += 1
                    else:
                        out_lbl = 0
                elif left != 0 and up == 0:
                    out_lbl = left
                elif left == 0 and up != 0:
                    out_lbl = up
                else:
                    if left < up:
                        out_lbl = left
                        parent[up] = left
                    elif left > up:
                        out_lbl = up
                        parent[left] = up
                    else:
                        out_lbl = left
                labels_pass1[idx] = out_lbl

    # UNION FIND RESOLVER
    for i in range(1, next_label):
        curr = i
        ptr = parent[curr]
        while True:
            p_ptr = parent[ptr]
            if p_ptr == ptr:
                break
            ptr = p_ptr
        parent[curr] = ptr

    # PASS 2
    labels_pass2 = np.zeros(total_pixels, dtype=np.uint16)
    for i in range(total_pixels):
        lbl = labels_pass1[i]
        if lbl != 0:
            labels_pass2[i] = parent[lbl]
            
    return labels_pass1, labels_pass2

def main(morph_in_path, out_dir):
    target_w, target_h = 1920, 1080
    expected_len = target_w * target_h
    
    try:
        with open(morph_in_path, 'r') as f:
            mask_stream = np.array([int(line.strip()) for line in f if line.strip()], dtype=np.uint8)
    except Exception as e:
        print(f"Error loading input mask: {e}")
        sys.exit(1)
        
    pass1_golden, pass2_golden = hw_accurate_ccl(mask_stream, target_w, target_h)
    
    p1_path = os.path.join(out_dir, "ccl_pass1_golden.txt")
    p2_path = os.path.join(out_dir, "ccl_pass2_golden.txt")
    
    with open(p1_path, 'w') as f1:
        for val in pass1_golden:
            f1.write(f"{val}\n")
            
    with open(p2_path, 'w') as f2:
        for val in pass2_golden:
            f2.write(f"{val}\n")
            
    print(f"CCL pre-processing complete. Max Label generated: {np.max(pass1_golden)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prep_ccl.py <actual_morph_out.txt> <out_data_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])