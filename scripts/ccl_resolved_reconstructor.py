#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

class CCLResolvedReconstructor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.height = 1080
        self.width = 1920
        
    def read_parent_table(self, parent_file: str) -> Dict[int, int]:
        parent_path = self.data_dir / parent_file
        parent_table = {}
        
        if not parent_path.exists():
            print(f"Warning: Parent table file not found: {parent_path}")
            return parent_table
        
        with open(parent_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        idx = int(parts[0])
                        parent = int(parts[1].strip('[]'))
                        parent_table[idx] = parent
                except ValueError:
                    continue
        
        return parent_table
    
    def resolve_labels(self, ccl_pass1: np.ndarray, parent_table: Dict[int, int]) -> np.ndarray:
        resolved = np.zeros_like(ccl_pass1)
        
        for i in range(ccl_pass1.shape[0]):
            for j in range(ccl_pass1.shape[1]):
                label = ccl_pass1[i, j]
                if label == 0:
                    resolved[i, j] = 0
                else:
                    root = label
                    visited = set()
                    while root in parent_table and root != parent_table[root] and root not in visited:
                        visited.add(root)
                        root = parent_table[root]
                    resolved[i, j] = root
                    
        return resolved
    
    def read_ccl_pass1(self, filename: str) -> np.ndarray:
        filepath = self.data_dir / filename
        labels = np.zeros((self.height, self.width), dtype=np.uint16)
        
        if not filepath.exists():
            print(f"Warning: Pass 1 file not found: {filepath}")
            return labels
        
        pixel_idx = 0
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    value = int(line)
                    row = pixel_idx // self.width
                    col = pixel_idx % self.width
                    if row < self.height:
                        labels[row, col] = value
                    pixel_idx += 1
                except ValueError:
                    continue
        
        return labels
    
    def reconstruct_from_rtl_outputs(self, pass1_filename: str, parent_filename: str) -> Tuple[np.ndarray, bool]:
        print(f"Reading CCL pass 1 from: {pass1_filename}")
        ccl_pass1 = self.read_ccl_pass1(pass1_filename)
        
        if np.max(ccl_pass1) == 0 and not (self.data_dir / pass1_filename).exists():
            print("Could not find or read valid CCL pass1 output file")
            return None, False
            
        print(f"Reading parent RAM from: {parent_filename}")
        parent_table = self.read_parent_table(parent_filename)
        
        if parent_table:
            print(f"Using parent table with {len(parent_table)} entries")
            resolved = self.resolve_labels(ccl_pass1, parent_table)
        else:
            print("Warning: No parent table found or parsed, using pass1 as resolved")
            resolved = ccl_pass1
            
        return resolved, True
    
    def save_resolved_labels(self, resolved: np.ndarray, output_file: str) -> None:
        output_path = self.data_dir / output_file
        
        with open(output_path, 'w') as f:
            f.write("# CCL Resolved Labels (After Union-Find Pass 2)\n")
            f.write(f"# Dimensions: {self.height}x{self.width}\n")
            f.write("# Format: Space-separated label IDs per line\n")
            
            unique_labels = len(np.unique(resolved)) - 1
            f.write(f"# Total Components: {unique_labels}\n\n")
            
            for row in resolved:
                row_str = "    " + "  ".join(f"{label:6d}" for label in row)
                f.write(row_str + "\n")
                
        print(f"CCL resolved labels saved to: {output_path}")
    
    def generate_ccl_resolved(self, pass1_filename: str, parent_filename: str, output_file: str) -> bool:
        print("=" * 70)
        print("CCL RESOLVED RECONSTRUCTION")
        print("=" * 70)
        
        resolved, success = self.reconstruct_from_rtl_outputs(pass1_filename, parent_filename)
        
        if not success or resolved is None:
            print("ERROR: Could not reconstruct CCL resolved from available files")
            return False
            
        print(f"Resolved labels shape: {resolved.shape}")
        print(f"Unique labels: {len(np.unique(resolved)) - 1}")
        
        self.save_resolved_labels(resolved, output_file)
        return True

def main():
    data_dir = "/users/epnyrk/Project/design/work/ProjectA/data"
    
    reconstructor = CCLResolvedReconstructor(data_dir)
    reconstructor.generate_ccl_resolved(
        pass1_filename="actual_ccl_labels.txt",
        parent_filename="rtl_ccl_resolved.txt",
        output_file="rtl_ccl_resolved_reconstructed.txt"
    )

if __name__ == "__main__":
    main()