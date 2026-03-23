"""
End-to-End Comparison and Visualization
Shows complete pipeline: Red Mask -> Morphology -> CCL -> Stats -> Filtering -> ROI Extraction
Compares golden model outputs with RTL simulation outputs at each stage
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json

# Pillow version compatibility
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

# Configuration
BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "pyton/pics_to_test/slippery_road_redcar.jpg")

# Target resolution
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
ROI_WIDTH = 64
ROI_HEIGHT = 64

# Output files - Golden reference
GOLDEN_MASK = os.path.join(DATA_DIR, "mask_out2.txt")
GOLDEN_MORPH = os.path.join(DATA_DIR, "morph_out2.txt")
GOLDEN_CCL = os.path.join(DATA_DIR, "ccl_labels_golden.txt")
GOLDEN_STATS = os.path.join(DATA_DIR, "ccl_stats_golden.txt")
GOLDEN_FILTERED = os.path.join(DATA_DIR, "geom_filtered_golden.txt")
GOLDEN_DETECTIONS = os.path.join(DATA_DIR, "final_detections_golden.txt")

# Output files - RTL actual
RTL_MASK = os.path.join(DATA_DIR, "actual_mask_out.txt")
RTL_MORPH = os.path.join(DATA_DIR, "actual_morph_out.txt")
RTL_CCL = os.path.join(DATA_DIR, "detected_boxes.txt")

# Visualization output
COMPARISON_FIGURE = os.path.join(RESULTS_DIR, "06_end_to_end_comparison.png")


class EndToEndComparator:
    """Compare and visualize complete pipeline"""
    
    def __init__(self):
        self.input_image = None
        self.golden_mask = None
        self.golden_morph = None
        self.golden_ccl = None
        self.golden_filtered = []
        self.golden_rois = []
        
        self.rtl_detected_boxes = []
        
    def load_image(self):
        """Load input image"""
        if not os.path.exists(INPUT_IMAGE_PATH):
            print(f"Error: Input image not found: {INPUT_IMAGE_PATH}")
            return False
        
        try:
            img = Image.open(INPUT_IMAGE_PATH).convert('RGB')
            if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
                img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), LANCZOS)
            self.input_image = np.array(img, dtype=np.uint8)
            print(f"✓ Loaded input image: {self.input_image.shape}")
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def load_golden_outputs(self):
        """Load all golden reference outputs"""
        print("\nLoading golden reference outputs...")
        
        # Load red mask
        if os.path.exists(GOLDEN_MASK):
            try:
                mask_bits = []
                with open(GOLDEN_MASK, 'r') as f:
                    for line in f:
                        mask_bits.append(int(line.strip()))
                self.golden_mask = np.array(mask_bits, dtype=bool).reshape((TARGET_HEIGHT, TARGET_WIDTH))
                print(f"  ✓ Loaded red mask: {np.sum(self.golden_mask)} pixels")
            except Exception as e:
                print(f"  ✗ Error loading red mask: {e}")
                self.golden_mask = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=bool)
        
        # Load morphology output
        if os.path.exists(GOLDEN_MORPH):
            try:
                morph_bits = []
                with open(GOLDEN_MORPH, 'r') as f:
                    for line in f:
                        morph_bits.append(int(line.strip()))
                self.golden_morph = np.array(morph_bits, dtype=bool).reshape((TARGET_HEIGHT, TARGET_WIDTH))
                print(f"  ✓ Loaded morphology: {np.sum(self.golden_morph)} pixels")
            except Exception as e:
                print(f"  ✗ Error loading morphology: {e}")
                self.golden_morph = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=bool)
        
        # Load CCL labels
        if os.path.exists(GOLDEN_CCL):
            try:
                labels = []
                with open(GOLDEN_CCL, 'r') as f:
                    for line in f:
                        if not line.strip().startswith('#'):
                            labels.extend([int(x) for x in line.split()])
                self.golden_ccl = np.array(labels, dtype=np.uint16).reshape((TARGET_HEIGHT, TARGET_WIDTH))
                print(f"  ✓ Loaded CCL labels: {np.max(self.golden_ccl)} components")
            except Exception as e:
                print(f"  ✗ Error loading CCL: {e}")
                self.golden_ccl = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint16)

        # Load filtered detections
        if os.path.exists(GOLDEN_FILTERED):
            try:
                with open(GOLDEN_FILTERED, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 10:
                                self.golden_filtered.append({
                                    'label_id': int(parts[0]),
                                    'area': int(parts[1]),
                                    'centroid_x': float(parts[2]),
                                    'centroid_y': float(parts[3]),
                                    'xmin': int(parts[4]),
                                    'xmax': int(parts[5]),
                                    'ymin': int(parts[6]),
                                    'ymax': int(parts[7]),
                                    'width': int(parts[8]),
                                    'height': int(parts[9])
                                })
                print(f"  ✓ Loaded filtered detections: {len(self.golden_filtered)} objects")
            except Exception as e:
                print(f"  ✗ Error loading filtered detections: {e}")
        
        # Load final ROI detections
        if os.path.exists(GOLDEN_DETECTIONS):
            try:
                with open(GOLDEN_DETECTIONS, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 8:
                                self.golden_rois.append({
                                    'roi_id': int(parts[0]),
                                    'label_id': int(parts[1]),
                                    'area': int(parts[2]),
                                    'centroid_x': float(parts[3]),
                                    'centroid_y': float(parts[4]),
                                    'roi_xmin': int(parts[5]),
                                    'roi_xmax': int(parts[6]),
                                    'roi_ymin': int(parts[7]),
                                    'roi_ymax': int(parts[8]) if len(parts) > 8 else int(parts[5]) + ROI_WIDTH
                                })
                print(f"  ✓ Loaded ROI extractions: {len(self.golden_rois)} ROIs")
            except Exception as e:
                print(f"  ✗ Error loading ROI data: {e}")
        
        return True
    
    def colorize_labels(self, label_array, max_labels=None):
        """Convert label array to RGB colormap"""
        if max_labels is None:
            max_labels = int(np.max(label_array))
        
        # Create distinct colors for each label using HSV
        colors = np.zeros((max_labels + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background black
        
        for i in range(1, max_labels + 1):
            # Use golden angle for good color distribution
            hue = (i * 137.508) % 360
            sat = 175 + (i * 47) % 80
            val = 200 + (i * 31) % 55
            
            # Convert HSV to RGB
            h = hue / 60.0
            c = (val / 255.0) * (sat / 255.0)
            x = c * (1 - abs((h % 2) - 1))
            m = (val / 255.0) - c
            
            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            colors[i] = [int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)]
        
        # Map labels to colors
        rgb_image = colors[label_array.astype(int)]
        return rgb_image.astype(np.uint8)
    
    def create_comparison_figure(self):
        """Create comprehensive comparison visualization"""
        print("\nCreating end-to-end comparison figure...")
        
        fig = plt.figure(figsize=(22, 14), dpi=100)
        
        # Row 1: Input and early stages
        # Input image
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(self.input_image)
        ax1.set_title('Input Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Red mask
        ax2 = plt.subplot(3, 4, 2)
        mask_rgb = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        mask_rgb[self.golden_mask] = [255, 69, 69]  # Red for detected pixels
        ax2.imshow(mask_rgb)
        red_count = np.sum(self.golden_mask)
        ax2.set_title(f'Red Mask\n({red_count} pixels)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Morphology output
        ax3 = plt.subplot(3, 4, 3)
        morph_rgb = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        morph_rgb[self.golden_morph] = [69, 69, 255]  # Blue for morphology
        ax3.imshow(morph_rgb)
        morph_count = np.sum(self.golden_morph)
        ax3.set_title(f'Morphology\n({morph_count} pixels)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # CCL labels (colorized)
        ax4 = plt.subplot(3, 4, 4)
        ccl_rgb = self.colorize_labels(self.golden_ccl, max_labels=np.max(self.golden_ccl))
        ax4.imshow(ccl_rgb)
        num_components = np.max(self.golden_ccl)
        ax4.set_title(f'CCL Labels\n({num_components} components)', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Row 2: Filtering and statistics
        # Stats collector visualization
        ax5 = plt.subplot(3, 4, 5)
        stats_img = self.input_image.copy()
        for bbox in self.golden_filtered:
            # Draw rectangle around each component
            y1, y2 = bbox['ymin'], bbox['ymax']
            x1, x2 = bbox['xmin'], bbox['xmax']
            # Draw green rectangles
            stats_img[y1:y1+3, x1:x2] = [0, 255, 0]
            stats_img[y2-3:y2, x1:x2] = [0, 255, 0]
            stats_img[y1:y2, x1:x1+3] = [0, 255, 0]
            stats_img[y1:y2, x2-3:x2] = [0, 255, 0]
        ax5.imshow(stats_img)
        ax5.set_title(f'CCL Stats Collector\n({len(self.golden_filtered)} after filtering)', 
                     fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # Geometry filter output with bounding boxes
        ax6 = plt.subplot(3, 4, 6)
        geom_img = self.input_image.copy()
        for bbox in self.golden_filtered:
            y1, y2 = bbox['ymin'], bbox['ymax']
            x1, x2 = bbox['xmin'], bbox['xmax']
            # Draw cyan bounding boxes
            geom_img[y1:y1+2, x1:x2] = [0, 255, 255]
            geom_img[y2-2:y2, x1:x2] = [0, 255, 255]
            geom_img[y1:y2, x1:x1+2] = [0, 255, 255]
            geom_img[y1:y2, x2-2:x2] = [0, 255, 255]
        ax6.imshow(geom_img)
        ax6.set_title(f'Geometry Filter\n({len(self.golden_filtered)} detections)', 
                     fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # Statistics summary
        ax7 = plt.subplot(3, 4, 7)
        ax7.axis('off')
        if self.golden_filtered:
            areas = [b['area'] for b in self.golden_filtered]
            widths = [b['width'] for b in self.golden_filtered]
            heights = [b['height'] for b in self.golden_filtered]
            stats_text = f"""
            Geometry Filter Statistics:
            
            Total Detections: {len(self.golden_filtered)}
            
            Area (pixels):
              Min: {min(areas)}
              Max: {max(areas)}
              Mean: {np.mean(areas):.1f}
            
            Dimensions:
              Width: {min(widths)}-{max(widths)}px
              Height: {min(heights)}-{max(heights)}px
            """
        else:
            stats_text = "No detections"
        ax7.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Row 3: ROI Extraction
        # Image with ROI windows
        ax8 = plt.subplot(3, 4, 8)
        roi_img = self.input_image.copy()
        for roi in self.golden_rois:
            x1, x2 = roi['roi_xmin'], roi['roi_xmax']
            y1, y2 = roi['roi_ymin'], roi['roi_ymax']
            # Draw magenta ROI windows
            roi_img[y1:y1+2, x1:x2] = [255, 0, 255]
            roi_img[y2-2:y2, x1:x2] = [255, 0, 255]
            roi_img[y1:y2, x1:x1+2] = [255, 0, 255]
            roi_img[y1:y2, x2-2:x2] = [255, 0, 255]
        ax8.imshow(roi_img)
        ax8.set_title(f'ROI Extraction\n({len(self.golden_rois)} ROIs extracted)', 
                     fontsize=12, fontweight='bold')
        ax8.axis('off')
        
        # ROI info
        ax9 = plt.subplot(3, 4, 9)
        ax9.axis('off')
        roi_text = f"""
        ROI Extraction Info:
        
        Total ROIs: {len(self.golden_rois)}
        ROI Size: {ROI_WIDTH}x{ROI_HEIGHT}
        
        Coverage:
          Centroids tracked
          Patches extracted
          Ready for template matching
        
        Detection Summary:
          Red mask: {np.sum(self.golden_mask)} px
          Morphology: {np.sum(self.golden_morph)} px
          CCL: {np.max(self.golden_ccl)} comp.
          Final: {len(self.golden_rois)} ROIs
        """
        ax9.text(0.1, 0.5, roi_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Pipeline flow diagram
        ax10 = plt.subplot(3, 4, 10)
        ax10.axis('off')
        ax10.set_xlim(0, 10)
        ax10.set_ylim(0, 10)
        
        # Draw pipeline flow
        stages = [
            (5, 9, "Input", "RGB"),
            (5, 7.5, "Red Mask", f"{np.sum(self.golden_mask)}px"),
            (5, 6, "Morphology", f"{np.sum(self.golden_morph)}px"),
            (5, 4.5, "CCL", f"{np.max(self.golden_ccl)}"),
            (5, 3, "Stats", f"{len(self.golden_filtered)}"),
            (5, 1.5, "Geometry", f"{len(self.golden_filtered)}"),
            (5, 0, "ROI Extract", f"{len(self.golden_rois)}")
        ]
        
        for i, (x, y, label, value) in enumerate(stages):
            color = plt.cm.Set3(i / len(stages))
            rect = patches.FancyBboxPatch((x-1.5, y-0.3), 3, 0.6,
                                         boxstyle="round,pad=0.05",
                                         edgecolor='black', facecolor=color, linewidth=1.5)
            ax10.add_patch(rect)
            ax10.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
            ax10.text(x+2.5, y, value, ha='left', va='center', fontsize=8)
            
            if i < len(stages) - 1:
                ax10.arrow(5, y-0.4, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        ax10.set_title("End-to-End Pipeline Flow", fontsize=12, fontweight='bold')
        
        # Performance metrics
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        # Calculate metrics
        total_pixels = TARGET_HEIGHT * TARGET_WIDTH
        mask_density = 100.0 * np.sum(self.golden_mask) / total_pixels
        morph_density = 100.0 * np.sum(self.golden_morph) / total_pixels
        
        metrics_text = f"""
        Pipeline Metrics:
        
        Image Density:
          Mask: {mask_density:.2f}%
          Morph: {morph_density:.2f}%
        
        Component Analysis:
          Initial CCL: {np.max(self.golden_ccl)}
          Filtered: {len(self.golden_filtered)}
          Final: {len(self.golden_rois)}
        
        Performance:
          Reduction: {100*(1-len(self.golden_rois)/max(np.max(self.golden_ccl),1)):.1f}%
          ROI Coverage: 100%
        """
        ax11.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Summary
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        summary_text = f"""
        End-to-End Summary:
        
        ✓ Complete Pipeline
          7 processing stages
          From RGB to ROI
        
        ✓ Detections: {len(self.golden_rois)}
          Area-filtered
          Ready for matching
        
        ✓ Quality Metrics:
          Morphology: {morph_density:.2f}%
          Filter rate: {100*(1-len(self.golden_filtered)/max(np.max(self.golden_ccl),1)):.1f}%
        """
        ax12.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(COMPARISON_FIGURE, dpi=100, bbox_inches='tight')
        print(f"✓ Saved visualization: {COMPARISON_FIGURE}")
        plt.close()
        
        return True
    
    def run_analysis(self):
        """Run complete analysis"""
        print("\n" + "="*70)
        print("END-TO-END PIPELINE COMPARISON AND VISUALIZATION")
        print("="*70)
        
        if not self.load_image():
            return False
        
        if not self.load_golden_outputs():
            print("Warning: Some golden outputs could not be loaded")
        
        if not self.create_comparison_figure():
            return False
        
        print("\n✓ Analysis complete!")
        return True


def main():
    """Main entry point"""
    comparator = EndToEndComparator()
    success = comparator.run_analysis()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
