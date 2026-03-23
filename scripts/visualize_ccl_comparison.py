"""
CCL Visualization and Comparison Tool
Visualizes CCL labels from both golden model and RTL simulation
Shows resolved labels with color mapping for better visibility
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configuration
BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
DATA_DIR = os.path.join(BASE_DIR, "data")

# Image parameters
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# Input files
CCL_GOLDEN_FILE = os.path.join(DATA_DIR, "ccl_labels_golden.txt")
CCL_RTL_FILE = os.path.join(DATA_DIR, "actual_ccl_labels.txt")
MORPH_GOLDEN_FILE = os.path.join(DATA_DIR, "morph_out2.txt")
MORPH_RTL_FILE = os.path.join(DATA_DIR, "actual_morph_out.txt")

# Output visualization files
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
CCL_GOLDEN_VIZ = os.path.join(VIZ_DIR, "ccl_golden_resolved.png")
CCL_RTL_VIZ = os.path.join(VIZ_DIR, "rtl_ccl_resolved_reconstructed.png")
CCL_COMPARISON_VIZ = os.path.join(VIZ_DIR, "ccl_comparison.png")

def create_viz_dir():
    """Create visualization directory if it doesn't exist"""
    if not os.path.exists(VIZ_DIR):
        os.makedirs(VIZ_DIR)
        print(f"✓ Created visualization directory: {VIZ_DIR}")

def load_ccl_labels(filepath):
    """Load CCL labels from text file"""
    print(f"\nLoading CCL labels from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return None
    
    labels = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint16)
    
    try:
        with open(filepath, 'r') as f:
            pixel_idx = 0
            for file_line in f:
                # Skip header comments
                line_stripped = file_line.strip()
                if line_stripped.startswith('#') or not line_stripped:
                    continue
                
                # Try parsing as single value per line (RTL format)
                tokens = line_stripped.split()
                
                if not tokens:
                    continue
                
                # Process tokens
                for token in tokens:
                    if pixel_idx >= TARGET_HEIGHT * TARGET_WIDTH:
                        break
                    try:
                        label_val = int(token)
                        row = pixel_idx // TARGET_WIDTH
                        col = pixel_idx % TARGET_WIDTH
                        labels[row, col] = label_val
                        pixel_idx += 1
                    except ValueError:
                        pass
                
                if pixel_idx >= TARGET_HEIGHT * TARGET_WIDTH:
                    break
        
        max_label = np.max(labels)
        unique_labels = len(np.unique(labels)) - 1  # Exclude background (0)
        
        print(f"  ✓ Loaded {TARGET_HEIGHT}x{TARGET_WIDTH} image")
        print(f"  ✓ Max label ID: {max_label}")
        print(f"  ✓ Unique components: {unique_labels}")
        
        return labels
    
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_colormap_visualization(labels, title, output_file):
    """Create a colored visualization of CCL labels"""
    if labels is None:
        return False
    
    print(f"\nGenerating visualization: {output_file}")
    
    max_label = np.max(labels)
    
    # Normalize for visualization
    if max_label > 0:
        normalized = (labels.astype(np.float32) / max_label * 255).astype(np.uint8)
    else:
        normalized = labels.astype(np.uint8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12), dpi=100)
    
    # Use HSV colormap for distinct colors
    import matplotlib
    cmap = matplotlib.colormaps['hsv']
    
    # Display with colormap
    im = ax.imshow(labels, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X Pixel', fontsize=12)
    ax.set_ylabel('Y Pixel', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Component Label ID', rotation=270, labelpad=20, fontsize=12)
    
    # Add grid for clarity
    ax.grid(True, which='major', color='white', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization with {max_label} components")
    return True

def create_difference_visualization(golden, rtl, output_file, morph_mask=None):
    """Create a difference visualization between golden and RTL
    If morph_mask provided, only compare pixels within morphology regions
    """
    if golden is None or rtl is None:
        print("✗ Cannot create difference visualization without both inputs")
        return False
    
    print(f"\nGenerating difference visualization: {output_file}")
    
    # Calculate differences - focus on morphology signal pixels
    if morph_mask is not None:
        # Only compare pixels within morphology region
        compare_region = morph_mask.astype(bool)
        match_within_region = (golden == rtl) & compare_region
        total_signal_pixels = np.sum(compare_region)
        matching_pixels = np.sum(match_within_region)
        error_pixels = total_signal_pixels - matching_pixels
        
        if total_signal_pixels > 0:
            accuracy = 100.0 * matching_pixels / total_signal_pixels
        else:
            accuracy = 100.0
        
        print(f"  Match Statistics (within morphology signal region):")
        print(f"    Signal region pixels: {total_signal_pixels:,}")
        print(f"    Matching pixels: {matching_pixels:,} ({accuracy:.2f}%)")
        print(f"    Error pixels: {error_pixels:,}")
    else:
        # Compare all pixels
        match_mask = (golden == rtl)
        total_pixels = match_mask.size
        matching_pixels = np.sum(match_mask)
        error_pixels = total_pixels - matching_pixels
        accuracy = 100.0 * matching_pixels / total_pixels
        
        print(f"  Match Statistics (all pixels):")
        print(f"    Matching pixels: {matching_pixels:,} / {total_pixels:,} ({accuracy:.2f}%)")
        print(f"    Error pixels: {error_pixels:,}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=100)
    
    from matplotlib import cm
    cmap = cm.get_cmap('hsv')
    
    # Golden
    im1 = axes[0].imshow(golden, cmap=cmap, interpolation='nearest')
    axes[0].set_title(f'Golden Model (Max Label: {np.max(golden)})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X Pixel')
    axes[0].set_ylabel('Y Pixel')
    
    # RTL
    im2 = axes[1].imshow(rtl, cmap=cmap, interpolation='nearest')
    axes[1].set_title(f'RTL Simulation (Max Label: {np.max(rtl)})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X Pixel')
    axes[1].set_ylabel('Y Pixel')
    
    # Difference (mismatch regions in red, background in gray)
    diff_display = np.ones((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8) * 128  # Gray background
    if morph_mask is not None:
        compare_region = morph_mask.astype(bool)
        match_within = (golden == rtl) & compare_region
        diff_display[match_within] = [0, 255, 0]  # Green for matches
        diff_display[~match_within & compare_region] = [255, 0, 0]  # Red for mismatches
    else:
        match_mask = (golden == rtl)
        diff_display[match_mask] = [0, 255, 0]  # Green for matches
        diff_display[~match_mask] = [255, 0, 0]  # Red for mismatches
    
    axes[2].imshow(diff_display)
    axes[2].set_title(f'Difference Map\nAccuracy: {accuracy:.2f}%', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X Pixel')
    axes[2].set_ylabel('Y Pixel')
    
    # Add legend
    green_patch = mpatches.Patch(color=(0, 1, 0), label='Match')
    red_patch = mpatches.Patch(color=(1, 0, 0), label='Mismatch')
    legend_items = [green_patch, red_patch]
    if morph_mask is not None:
        gray_patch = mpatches.Patch(color=(0.5, 0.5, 0.5), label='Background (ignored)')
        legend_items.append(gray_patch)
    axes[2].legend(handles=legend_items, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved comparison visualization")
    return True

def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("CCL VISUALIZATION AND COMPARISON TOOL")
    print("="*70)
    
    # Create visualization directory
    create_viz_dir()
    
    # Load golden and RTL CCL labels
    golden_ccl = load_ccl_labels(CCL_GOLDEN_FILE)
    rtl_ccl = load_ccl_labels(CCL_RTL_FILE)
    
    # Load morphology masks for comparison filtering
    morph_golden = load_ccl_labels(MORPH_GOLDEN_FILE)
    
    # Create visualizations
    if golden_ccl is not None:
        create_colormap_visualization(
            golden_ccl,
            "Golden Model - CCL Resolved Labels (After Union-Find Pass 2)",
            CCL_GOLDEN_VIZ
        )
    
    if rtl_ccl is not None:
        create_colormap_visualization(
            rtl_ccl,
            "RTL Simulation - CCL Labels",
            CCL_RTL_VIZ
        )
    
    # Create comparison (focus on morphology signal region)
    if golden_ccl is not None and rtl_ccl is not None:
        create_difference_visualization(golden_ccl, rtl_ccl, CCL_COMPARISON_VIZ, morph_mask=morph_golden)
    
    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print(f"  Output directory: {VIZ_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
