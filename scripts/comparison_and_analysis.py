"""
Comparison and Analysis Tool
Compares RTL simulation results with golden model outputs
Provides detailed statistics, differences, and visualizations
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image
import struct

# Paths
BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "pyton/pics_to_test/slippery_road_redcar.jpg")

# Image parameters
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Reference files (from golden model)
MASK_OUT_GOLDEN = os.path.join(DATA_DIR, "mask_out2.txt")
MORPH_OUT_GOLDEN = os.path.join(DATA_DIR, "morph_out2.txt")

# RTL simulation output files
MASK_OUT_RTL = os.path.join(DATA_DIR, "actual_mask_out.txt")
MORPH_OUT_RTL = os.path.join(DATA_DIR, "actual_morph_out.txt")


class ComparisonAnalyzer:
    """Analyzes differences between RTL and golden model outputs"""
    
    def __init__(self):
        """Initialize analyzer"""
        self.mask_golden = None
        self.mask_rtl = None
        self.morph_golden = None
        self.morph_rtl = None
        
        self.mask_diff = None
        self.morph_diff = None
        
        self.mask_stats = {}
        self.morph_stats = {}
        
        # Load input image for visualization
        self.input_image = self.load_input_image()
    
    def load_input_image(self):
        """Load the input RGB image"""
        try:
            img = Image.open(INPUT_IMAGE_PATH)
            if img.size != (IMG_WIDTH, IMG_HEIGHT):
                img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
            return np.array(img)
        except:
            print("Warning: Could not load input image")
            return None
    
    def load_binary_file(self, filepath, num_pixels):
        """Load binary data from file (each line is '0' or '1')"""
        try:
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line in ['0', '1']:
                        data.append(int(line))
                    if len(data) >= num_pixels:
                        break
            
            if len(data) != num_pixels:
                print(f"Warning: Expected {num_pixels} pixels, got {len(data)}")
                # Pad with zeros if needed
                data.extend([0] * (num_pixels - len(data)))
            
            return np.array(data, dtype=bool).reshape((IMG_HEIGHT, IMG_WIDTH))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_all_data(self):
        """Load all comparison data"""
        print("\n" + "="*70)
        print("LOADING COMPARISON DATA")
        print("="*70)
        
        num_pixels = IMG_HEIGHT * IMG_WIDTH
        
        # Load mask data
        print("\nLoading mask data...")
        self.mask_golden = self.load_binary_file(MASK_OUT_GOLDEN, num_pixels)
        self.mask_rtl = self.load_binary_file(MASK_OUT_RTL, num_pixels)
        
        if self.mask_golden is not None:
            print(f"  ✓ Golden mask: {np.sum(self.mask_golden)} pixels")
        if self.mask_rtl is not None:
            print(f"  ✓ RTL mask: {np.sum(self.mask_rtl)} pixels")
        
        # Load morphology data
        print("\nLoading morphology data...")
        self.morph_golden = self.load_binary_file(MORPH_OUT_GOLDEN, num_pixels)
        self.morph_rtl = self.load_binary_file(MORPH_OUT_RTL, num_pixels)
        
        if self.morph_golden is not None:
            print(f"  ✓ Golden morph: {np.sum(self.morph_golden)} pixels")
        if self.morph_rtl is not None:
            print(f"  ✓ RTL morph: {np.sum(self.morph_rtl)} pixels")
        
        return (self.mask_golden is not None and self.mask_rtl is not None and
                self.morph_golden is not None and self.morph_rtl is not None)
    
    def compute_differences(self):
        """Compute differences between RTL and golden outputs"""
        print("\n" + "="*70)
        print("COMPUTING DIFFERENCES")
        print("="*70)
        
        if self.mask_golden is not None and self.mask_rtl is not None:
            self.mask_diff = np.bitwise_xor(self.mask_golden, self.mask_rtl)
            self.compute_mask_statistics()
        
        if self.morph_golden is not None and self.morph_rtl is not None:
            self.morph_diff = np.bitwise_xor(self.morph_golden, self.morph_rtl)
            self.compute_morph_statistics()
    
    def compute_mask_statistics(self):
        """Compute statistics for mask comparison"""
        total_pixels = IMG_HEIGHT * IMG_WIDTH
        
        golden_ones = np.sum(self.mask_golden)
        rtl_ones = np.sum(self.mask_rtl)
        diff_pixels = np.sum(self.mask_diff)
        accuracy = 100.0 * (1.0 - diff_pixels / total_pixels)
        
        self.mask_stats = {
            'total_pixels': total_pixels,
            'golden_ones': golden_ones,
            'rtl_ones': rtl_ones,
            'diff_pixels': diff_pixels,
            'accuracy': accuracy,
            'golden_percent': 100.0 * golden_ones / total_pixels,
            'rtl_percent': 100.0 * rtl_ones / total_pixels,
        }
        
        print("\nMask Stage Comparison:")
        print(f"  Golden pixels: {golden_ones} ({self.mask_stats['golden_percent']:.2f}%)")
        print(f"  RTL pixels:    {rtl_ones} ({self.mask_stats['rtl_percent']:.2f}%)")
        print(f"  Differences:   {diff_pixels} pixels")
        print(f"  Accuracy:      {accuracy:.4f}%")
    
    def compute_morph_statistics(self):
        """Compute statistics for morphology comparison"""
        total_pixels = IMG_HEIGHT * IMG_WIDTH
        
        golden_ones = np.sum(self.morph_golden)
        rtl_ones = np.sum(self.morph_rtl)
        diff_pixels = np.sum(self.morph_diff)
        accuracy = 100.0 * (1.0 - diff_pixels / total_pixels)
        
        self.morph_stats = {
            'total_pixels': total_pixels,
            'golden_ones': golden_ones,
            'rtl_ones': rtl_ones,
            'diff_pixels': diff_pixels,
            'accuracy': accuracy,
            'golden_percent': 100.0 * golden_ones / total_pixels,
            'rtl_percent': 100.0 * rtl_ones / total_pixels,
        }
        
        print("\nMorphology Stage Comparison:")
        print(f"  Golden pixels: {golden_ones} ({self.morph_stats['golden_percent']:.2f}%)")
        print(f"  RTL pixels:    {rtl_ones} ({self.morph_stats['rtl_percent']:.2f}%)")
        print(f"  Differences:   {diff_pixels} pixels")
        print(f"  Accuracy:      {accuracy:.4f}%")
    
    def print_summary(self):
        """Print summary report"""
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        if self.mask_stats:
            print("\n[MASK STAGE]")
            print(f"  Total pixels processed: {self.mask_stats['total_pixels']:,}")
            print(f"  Golden match pixels:    {self.mask_stats['golden_ones']:,}")
            print(f"  RTL match pixels:       {self.mask_stats['rtl_ones']:,}")
            print(f"  Mismatched pixels:      {self.mask_stats['diff_pixels']:,}")
            print(f"  Accuracy:               {self.mask_stats['accuracy']:.6f}%")
        
        if self.morph_stats:
            print("\n[MORPHOLOGY STAGE]")
            print(f"  Total pixels processed: {self.morph_stats['total_pixels']:,}")
            print(f"  Golden match pixels:    {self.morph_stats['golden_ones']:,}")
            print(f"  RTL match pixels:       {self.morph_stats['rtl_ones']:,}")
            print(f"  Mismatched pixels:      {self.morph_stats['diff_pixels']:,}")
            print(f"  Accuracy:               {self.morph_stats['accuracy']:.6f}%")
        
        print("\n" + "="*70)
    
    def visualize_mask_stage(self):
        """Create visualization for mask stage comparison"""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 0: Input Image and Masks
        ax0 = fig.add_subplot(gs[0, 0])
        if self.input_image is not None:
            ax0.imshow(self.input_image)
        ax0.set_title("Input RGB Image", fontsize=12, fontweight='bold')
        ax0.axis('off')
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(self.mask_golden, cmap='gray')
        ax1.set_title("Golden Model Mask", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(self.mask_rtl, cmap='gray')
        ax2.set_title("RTL Simulation Mask", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 3])
        diff_colored = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        diff_colored[self.mask_diff] = [1, 0, 0]  # Red for differences
        ax3.imshow(diff_colored)
        ax3.set_title(f"Differences ({self.mask_stats['diff_pixels']} pixels)", 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Row 1: Difference visualization
        ax4 = fig.add_subplot(gs[1, 0])
        false_positives = np.bitwise_and(~self.mask_golden, self.mask_rtl)
        false_negatives = np.bitwise_and(self.mask_golden, ~self.mask_rtl)
        
        fp_count = np.sum(false_positives)
        fn_count = np.sum(false_negatives)
        
        ax4.bar(['False Positives', 'False Negatives'], [fp_count, fn_count],
               color=['#2ecc71', '#e74c3c'])
        ax4.set_ylabel('Pixel Count', fontsize=11)
        ax4.set_title('Error Breakdown', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        fp_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        fp_img[false_positives] = [0, 1, 0]  # Green for FP
        ax5.imshow(fp_img)
        ax5.set_title(f'False Positives: {fp_count}', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        fn_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        fn_img[false_negatives] = [1, 0, 0]  # Red for FN
        ax6.imshow(fn_img)
        ax6.set_title(f'False Negatives: {fn_count}', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 3])
        accuracy_data = [self.mask_stats['accuracy'], 100 - self.mask_stats['accuracy']]
        colors = ['#27ae60', '#e74c3c']
        wedges, texts, autotexts = ax7.pie(accuracy_data, labels=['Match', 'Mismatch'],
                                            autopct='%1.4f%%', colors=colors, startangle=90)
        ax7.set_title(f"Accuracy: {self.mask_stats['accuracy']:.4f}%", 
                     fontsize=12, fontweight='bold')
        
        # Row 2: Statistics table
        ax8 = fig.add_subplot(gs[2, :])
        ax8.axis('off')
        
        stats_text = f"""
        MASK STAGE DETAILED STATISTICS
        
        Golden Model:         {self.mask_stats['golden_ones']:,} pixels ({self.mask_stats['golden_percent']:.2f}%)
        RTL Simulation:       {self.mask_stats['rtl_ones']:,} pixels ({self.mask_stats['rtl_percent']:.2f}%)
        False Positives (FP): {fp_count} pixels (RTL=1, Golden=0)
        False Negatives (FN): {fn_count} pixels (RTL=0, Golden=1)
        Total Mismatches:     {self.mask_stats['diff_pixels']} pixels
        Overall Accuracy:     {self.mask_stats['accuracy']:.6f}%
        """
        
        ax8.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.suptitle("RED MASK STAGE - Golden Model vs RTL Comparison", 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save figure
        output_file = os.path.join(RESULTS_DIR, "01_mask_comparison.png")
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"\n✓ Saved mask comparison to: {output_file}")
        plt.close()
    
    def visualize_morph_stage(self):
        """Create visualization for morphology stage comparison"""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 0: Morphology outputs
        ax0 = fig.add_subplot(gs[0, 0])
        if self.mask_rtl is not None:
            ax0.imshow(self.mask_rtl, cmap='gray')
        ax0.set_title("Input (Mask Output)", fontsize=12, fontweight='bold')
        ax0.axis('off')
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(self.morph_golden, cmap='gray')
        ax1.set_title("Golden Morphology", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(self.morph_rtl, cmap='gray')
        ax2.set_title("RTL Morphology", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 3])
        diff_colored = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        diff_colored[self.morph_diff] = [1, 0, 0]  # Red for differences
        ax3.imshow(diff_colored)
        ax3.set_title(f"Differences ({self.morph_stats['diff_pixels']} pixels)", 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Row 1: Error analysis
        ax4 = fig.add_subplot(gs[1, 0])
        false_positives = np.bitwise_and(~self.morph_golden, self.morph_rtl)
        false_negatives = np.bitwise_and(self.morph_golden, ~self.morph_rtl)
        
        fp_count = np.sum(false_positives)
        fn_count = np.sum(false_negatives)
        
        ax4.bar(['False Positives', 'False Negatives'], [fp_count, fn_count],
               color=['#3498db', '#e67e22'])
        ax4.set_ylabel('Pixel Count', fontsize=11)
        ax4.set_title('Error Breakdown', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        fp_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        fp_img[false_positives] = [0, 0, 1]  # Blue for FP
        ax5.imshow(fp_img)
        ax5.set_title(f'False Positives: {fp_count}', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        fn_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        fn_img[false_negatives] = [1, 0.5, 0]  # Orange for FN
        ax6.imshow(fn_img)
        ax6.set_title(f'False Negatives: {fn_count}', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 3])
        accuracy_data = [self.morph_stats['accuracy'], 100 - self.morph_stats['accuracy']]
        colors = ['#27ae60', '#e74c3c']
        wedges, texts, autotexts = ax7.pie(accuracy_data, labels=['Match', 'Mismatch'],
                                            autopct='%1.4f%%', colors=colors, startangle=90)
        ax7.set_title(f"Accuracy: {self.morph_stats['accuracy']:.4f}%", 
                     fontsize=12, fontweight='bold')
        
        # Row 2: Statistics
        ax8 = fig.add_subplot(gs[2, :])
        ax8.axis('off')
        
        stats_text = f"""
        MORPHOLOGY STAGE DETAILED STATISTICS
        
        Golden Model:         {self.morph_stats['golden_ones']:,} pixels ({self.morph_stats['golden_percent']:.2f}%)
        RTL Simulation:       {self.morph_stats['rtl_ones']:,} pixels ({self.morph_stats['rtl_percent']:.2f}%)
        False Positives (FP): {fp_count} pixels (RTL=1, Golden=0)
        False Negatives (FN): {fn_count} pixels (RTL=0, Golden=1)
        Total Mismatches:     {self.morph_stats['diff_pixels']} pixels
        Overall Accuracy:     {self.morph_stats['accuracy']:.6f}%
        """
        
        ax8.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightblue', alpha=0.5))
        
        plt.suptitle("MORPHOLOGY STAGE - Golden Model vs RTL Comparison", 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save figure
        output_file = os.path.join(RESULTS_DIR, "02_morph_comparison.png")
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"✓ Saved morph comparison to: {output_file}")
        plt.close()
    
    def visualize_pipeline_flow(self):
        """Create a visualization of the entire pipeline flow"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("ROAD SIGN DETECTOR - PROCESSING PIPELINE FLOW", 
                    fontsize=14, fontweight='bold')
        
        # Stage 1: Input
        ax = axes[0, 0]
        if self.input_image is not None:
            ax.imshow(self.input_image)
        ax.set_title("Stage 0: Input Image", fontweight='bold')
        ax.axis('off')
        
        # Stage 2: Red Mask
        ax = axes[0, 1]
        mask_colored = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        mask_colored[self.mask_golden] = [1, 0, 0]  # Red pixels
        if self.input_image is not None:
            background = self.input_image.astype(float) / 255.0
            background = 0.3 * background  # Darken background
            mask_colored[self.mask_golden] = [1, 0, 0]  # Overlay red
            ax.imshow(background)
            ax.imshow(mask_colored)
        else:
            ax.imshow(self.mask_golden, cmap='gray')
        ax.set_title("Stage 1: Red Mask Detection", fontweight='bold')
        ax.axis('off')
        
        # Stage 3: Morphology
        ax = axes[0, 2]
        morph_colored = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        morph_colored[self.morph_golden] = [0, 1, 0]  # Green pixels
        if self.input_image is not None:
            background = self.input_image.astype(float) / 255.0
            background = 0.3 * background
            morph_colored[self.morph_golden] = [0, 1, 0]
            ax.imshow(background)
            ax.imshow(morph_colored)
        else:
            ax.imshow(self.morph_golden, cmap='gray')
        ax.set_title("Stage 2: Morphology Filter", fontweight='bold')
        ax.axis('off')
        
        # Stage 4: Comparison - Mask
        ax = axes[1, 0]
        if self.mask_rtl is not None and self.mask_golden is not None:
            comparison = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
            comparison[self.mask_golden] = [0, 1, 0]  # Green = golden
            comparison[self.mask_rtl] = [1, 0, 0]  # Red = RTL
            comparison[np.bitwise_and(self.mask_golden, self.mask_rtl)] = [1, 1, 0]  # Yellow = match
            ax.imshow(comparison)
        ax.set_title(f"Mask Accuracy: {self.mask_stats['accuracy']:.4f}%", fontweight='bold')
        ax.axis('off')
        
        # Comparison - Morphology
        ax = axes[1, 1]
        if self.morph_rtl is not None and self.morph_golden is not None:
            comparison = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
            comparison[self.morph_golden] = [0, 1, 0]  # Green = golden
            comparison[self.morph_rtl] = [1, 0, 0]  # Red = RTL
            comparison[np.bitwise_and(self.morph_golden, self.morph_rtl)] = [1, 1, 0]  # Yellow = match
            ax.imshow(comparison)
        ax.set_title(f"Morph Accuracy: {self.morph_stats['accuracy']:.4f}%", fontweight='bold')
        ax.axis('off')
        
        # Statistics summary
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
COMPARISON RESULTS SUMMARY

Red Mask:
  Golden: {self.mask_stats['golden_ones']:,} pixels
  RTL:    {self.mask_stats['rtl_ones']:,} pixels
  Error:  {self.mask_stats['diff_pixels']} pixels
  Acc:    {self.mask_stats['accuracy']:.4f}%

Morphology:
  Golden: {self.morph_stats['golden_ones']:,} pixels
  RTL:    {self.morph_stats['rtl_ones']:,} pixels
  Error:  {self.morph_stats['diff_pixels']} pixels
  Acc:    {self.morph_stats['accuracy']:.4f}%
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round',
               facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        output_file = os.path.join(RESULTS_DIR, "03_pipeline_flow.png")
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"✓ Saved pipeline flow to: {output_file}")
        plt.close()
    
    def run_analysis(self):
        """Run complete analysis"""
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " COMPARISON AND ANALYSIS TOOL ".center(68) + "║")
        print("╚" + "="*68 + "╝")
        
        # Create results directory if needed
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        # Load all data
        if not self.load_all_data():
            print("✗ Failed to load comparison data")
            return False
        
        # Compute differences
        self.compute_differences()
        
        # Print summary
        self.print_summary()
        
        # Generate visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        self.visualize_mask_stage()
        self.visualize_morph_stage()
        self.visualize_pipeline_flow()
        
        print("\n✓ Analysis complete!")
        print(f"  Results saved to: {RESULTS_DIR}")
        
        return True


def main():
    """Main entry point"""
    analyzer = ComparisonAnalyzer()
    success = analyzer.run_analysis()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
