import sys
import os
import numpy as np

def verify_pass(golden_path, rtl_path, pass_name, expected_len):
    try:
        with open(golden_path, 'r') as fg:
            golden_data = np.array([int(line.strip()) for line in fg if line.strip()], dtype=np.uint16)
        with open(rtl_path, 'r') as fr:
            rtl_data = np.array([int(line.strip()) for line in fr if line.strip()], dtype=np.uint16)
    except Exception as e:
        print(f"Error reading files for {pass_name}: {e}")
        return False
        
    if len(golden_data) != expected_len or len(rtl_data) != expected_len:
        print(f"Error: Length mismatch in {pass_name}.")
        return False
        
    diff = (golden_data != rtl_data)
    errors = int(np.sum(diff))
    
    print(f"{pass_name} Mismatches: {errors}")
    return errors == 0

def main(golden_p1, rtl_p1, golden_p2, rtl_p2, out_dir):
    target_w, target_h = 1920, 1080
    expected_len = target_w * target_h
    
    print("----------------------------------------")
    print("CCL VERIFICATION RESULTS (FHD)")
    print("----------------------------------------")
    
    p1_pass = verify_pass(golden_p1, rtl_p1, "PASS 1 (Raw Labels)", expected_len)
    p2_pass = verify_pass(golden_p2, rtl_p2, "PASS 2 (Resolved Labels)", expected_len)
    
    overall_status = "PASS" if (p1_pass and p2_pass) else "FAIL"
    print("----------------------------------------")
    print(f"Overall Status: {overall_status}")
    print("----------------------------------------")
    
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, "ccl_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"STAGE: CCL\n")
        f.write(f"PASS_1_OK: {p1_pass}\n")
        f.write(f"PASS_2_OK: {p2_pass}\n")
        f.write(f"STATUS: {overall_status}\n")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python verify_ccl.py <gold_p1> <rtl_p1> <gold_p2> <rtl_p2> <out_results_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])