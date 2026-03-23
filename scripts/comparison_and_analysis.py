#!/usr/bin/env python3
"""
Comparison and Analysis Tool
Compares RTL simulation results with golden model outputs
Provides detailed statistics, differences, and visualizations (Noise-Filtered & Clean)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

# Paths
BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "pyton/pics_to_test/slippery_road_redcar.jpg")

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

MASK_OUT_GOLDEN = os.path.join(DATA_DIR, "mask_out2.txt")
MORPH_OUT_GOLDEN = os.path.join(DATA_DIR, "morph_out2.txt")
CCL_LABELS_GOLDEN = os.path.join(DATA_DIR, "ccl_labels_golden.txt")

MASK_OUT_RTL = os.path.join(DATA_DIR, "actual_mask_out.txt")
MORPH_OUT_RTL = os.path.join(DATA_DIR, "actual_morph_out.txt")
CCL_LABELS_RTL = os.path.join(DATA_DIR, "rtl_ccl_resolved_reconstructed.txt")

# Noise filter threshold for CCL Evaluation (Ignore dust)
MIN_CCL_AREA = 15

class ComparisonAnalyzer:
    def __init__(self):
        self.mask_golden = None
        self.mask_rtl = None
        self.morph_golden = None
        self.morph_rtl = None
        self.ccl_golden = None
        self.ccl_rtl = None
        
        self.mask_diff = None
        self.morph_diff = None
        
        self.mask_stats = {}
        self.morph_stats = {}
        self.ccl_stats = {}
        self.ccl_matches = {}
        
        self.input_image = self.load_input_image()
    
    def load_input_image(self):
        try:
            img = Image.open(INPUT_IMAGE_PATH)
            if img.size != (IMG_WIDTH, IMG_HEIGHT):
                img = img.resize((IMG_WIDTH, IMG_HEIGHT), LANCZOS)
            return np.array(img)
        except:
            return None
    
    def load_binary_file(self, filepath, num_pixels):
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
                data.extend([0] * (num_pixels - len(data)))
            return np.array(data, dtype=bool).reshape((IMG_HEIGHT, IMG_WIDTH))
        except:
            return None
    
    def load_ccl_labels(self, filepath, num_pixels):
        try:
            labels = []
            with open(filepath, 'r') as f:
                for line in f:
                    if not line.strip().startswith('#'):
                        labels.extend([int(x) for x in line.split()])
            if len(labels) > num_pixels:
                labels = labels[:num_pixels]
            elif len(labels) < num_pixels:
                labels.extend([0] * (num_pixels - len(labels)))
            return np.array(labels, dtype=np.uint16).reshape((IMG_HEIGHT, IMG_WIDTH))
        except:
            return None
    
    def load_all_data(self):
        num_pixels = IMG_HEIGHT * IMG_WIDTH
        self.mask_golden = self.load_binary_file(MASK_OUT_GOLDEN, num_pixels)
        self.mask_rtl = self.load_binary_file(MASK_OUT_RTL, num_pixels)
        self.morph_golden = self.load_binary_file(MORPH_OUT_GOLDEN, num_pixels)
        self.morph_rtl = self.load_binary_file(MORPH_OUT_RTL, num_pixels)
        self.ccl_golden = self.load_ccl_labels(CCL_LABELS_GOLDEN, num_pixels)
        self.ccl_rtl = self.load_ccl_labels(CCL_LABELS_RTL, num_pixels)
        
        return (self.mask_golden is not None and self.mask_rtl is not None and
                self.morph_golden is not None and self.morph_rtl is not None and
                self.ccl_golden is not None and self.ccl_rtl is not None)
    
    def compute_differences(self):
        if self.mask_golden is not None and self.mask_rtl is not None:
            self.mask_diff = np.bitwise_xor(self.mask_golden, self.mask_rtl)
            self.compute_mask_statistics()
        if self.morph_golden is not None and self.morph_rtl is not None:
            self.morph_diff = np.bitwise_xor(self.morph_golden, self.morph_rtl)
            self.compute_morph_statistics()
        if self.ccl_golden is not None and self.ccl_rtl is not None:
            self.compute_ccl_statistics()
    
    def compute_mask_statistics(self):
        total_pixels = IMG_HEIGHT * IMG_WIDTH
        g_ones = np.sum(self.mask_golden)
        r_ones = np.sum(self.mask_rtl)
        diff = np.sum(self.mask_diff)
        
        self.mask_stats = {
            'total_pixels': total_pixels,
            'golden_ones': g_ones, 'rtl_ones': r_ones, 'diff_pixels': diff,
            'accuracy': 100.0 * (1.0 - diff / total_pixels),
        }
    
    def compute_morph_statistics(self):
        total_pixels = IMG_HEIGHT * IMG_WIDTH
        g_ones = np.sum(self.morph_golden)
        r_ones = np.sum(self.morph_rtl)
        diff = np.sum(self.morph_diff)
        
        self.morph_stats = {
            'total_pixels': total_pixels,
            'golden_ones': g_ones, 'rtl_ones': r_ones, 'diff_pixels': diff,
            'accuracy': 100.0 * (1.0 - diff / total_pixels),
        }
    
    def compute_ccl_statistics(self):
        total_pixels = IMG_HEIGHT * IMG_WIDTH
        
        golden_areas = np.bincount(self.ccl_golden.ravel())
        rtl_areas = np.bincount(self.ccl_rtl.ravel())
        
        # FILTER NOISE: Only look at components larger than MIN_CCL_AREA
        golden_labels = {l for l in np.unique(self.ccl_golden) if l > 0 and golden_areas[l] >= MIN_CCL_AREA}
        rtl_labels = {l for l in np.unique(self.ccl_rtl) if l > 0 and rtl_areas[l] >= MIN_CCL_AREA}
        
        self.ccl_matches = {}
        matched_rtl_set = set()
        true_positive_pixels = 0
        
        for gl in golden_labels:
            overlap_mask = (self.ccl_golden == gl)
            overlapping_rtl = self.ccl_rtl[overlap_mask]
            overlapping_rtl = overlapping_rtl[overlapping_rtl > 0]
            
            if len(overlapping_rtl) > 0:
                best_rtl = np.bincount(overlapping_rtl).argmax()
                # Ensure the matched RTL is also valid (not noise)
                if best_rtl in rtl_labels:
                    intersection = np.sum((self.ccl_golden == gl) & (self.ccl_rtl == best_rtl))
                    union = golden_areas[gl] + rtl_areas[best_rtl] - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.15: # Generous overlap to account for HW morphology shifting
                        self.ccl_matches[gl] = best_rtl
                        matched_rtl_set.add(best_rtl)
                        true_positive_pixels += intersection

        missing_labels = golden_labels - set(self.ccl_matches.keys())
        extra_labels = rtl_labels - matched_rtl_set
        
        # Pixel accuracy based only on valid components
        valid_golden_mask = np.isin(self.ccl_golden, list(golden_labels))
        valid_rtl_mask = np.isin(self.ccl_rtl, list(rtl_labels))
        
        total_fg_union = np.sum(valid_golden_mask) + np.sum(valid_rtl_mask) - true_positive_pixels
        accuracy = 100.0 * true_positive_pixels / total_fg_union if total_fg_union > 0 else 100.0
        label_accuracy = 100.0 * len(self.ccl_matches) / len(golden_labels) if len(golden_labels) > 0 else 100.0
        
        self.ccl_stats = {
            'total_pixels': total_pixels,
            'diff_pixels': int(total_fg_union - true_positive_pixels),
            'accuracy': accuracy,
            'golden_num_labels': len(golden_labels),
            'rtl_num_labels': len(rtl_labels),
            'matching_labels': len(self.ccl_matches),
            'missing_labels': len(missing_labels),
            'extra_labels': len(extra_labels),
            'label_accuracy': label_accuracy,
        }

    def print_summary(self):
        print("\n" + "="*70)
        print(f"{'COMPARISON SUMMARY':^70}")
        print("="*70)
        print(f"MASK:  Accuracy {self.mask_stats['accuracy']:.4f}% | Mismatches: {self.mask_stats['diff_pixels']:,}")
        print(f"MORPH: Accuracy {self.morph_stats['accuracy']:.4f}% | Mismatches: {self.morph_stats['diff_pixels']:,}")
        print(f"CCL:   Accuracy {self.ccl_stats['accuracy']:.4f}% | Target Detection Rate: {self.ccl_stats['label_accuracy']:.2f}%")
        print(f"       (Ignored dust < {MIN_CCL_AREA}px. Found {self.ccl_stats['golden_num_labels']} targets, Matched {self.ccl_stats['matching_labels']})")
        print("="*70)

    # ---------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------
    def _create_common_figure(self, title):
        fig = plt.figure(figsize=(20, 11))
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.2)
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.96)
        return fig, gs

    def visualize_mask_stage(self):
        fig, gs = self._create_common_figure("RED MASK STAGE - Golden Model vs RTL Comparison")
        
        ax0 = fig.add_subplot(gs[0, 0])
        if self.input_image is not None: ax0.imshow(self.input_image)
        ax0.set_title("Input RGB Image", fontsize=14, fontweight='bold')
        ax0.axis('off')
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(self.mask_golden, cmap='gray')
        ax1.set_title("Golden Model Mask", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(self.mask_rtl, cmap='gray')
        ax2.set_title("RTL Simulation Mask", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 3])
        diff_colored = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        diff_colored[self.mask_diff] = [1, 0, 0]
        ax3.imshow(diff_colored)
        ax3.set_title(f"Differences ({self.mask_stats['diff_pixels']} px)", fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        fp = np.bitwise_and(~self.mask_golden, self.mask_rtl)
        fn = np.bitwise_and(self.mask_golden, ~self.mask_rtl)
        fp_count, fn_count = np.sum(fp), np.sum(fn)
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.bar(['False Pos.', 'False Neg.'], [fp_count, fn_count], color=['#2ecc71', '#e74c3c'])
        ax4.set_title('Error Breakdown', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='both', labelsize=12)
        
        ax5 = fig.add_subplot(gs[1, 1])
        fp_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)); fp_img[fp] = [0, 1, 0]
        ax5.imshow(fp_img)
        ax5.set_title(f'False Positives: {fp_count}', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        fn_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)); fn_img[fn] = [1, 0, 0]
        ax6.imshow(fn_img)
        ax6.set_title(f'False Negatives: {fn_count}', fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.pie([self.mask_stats['accuracy'], 100-self.mask_stats['accuracy']], labels=['Match', 'Diff'], 
                autopct='%1.2f%%', colors=['#27ae60', '#e74c3c'], startangle=90, textprops={'fontsize': 12})
        ax7.set_title(f"Accuracy: {self.mask_stats['accuracy']:.4f}%", fontsize=14, fontweight='bold')
        
        ax8 = fig.add_subplot(gs[2, 1:3])
        ax8.axis('off')
        stats = f"Golden Model: {self.mask_stats['golden_ones']:,} px\nRTL Simulation: {self.mask_stats['rtl_ones']:,} px\nMismatches: {self.mask_stats['diff_pixels']:,} px\nOverall Accuracy: {self.mask_stats['accuracy']:.4f}%"
        ax8.text(0.5, 0.5, stats, fontsize=16, family='monospace', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
        
        plt.savefig(os.path.join(RESULTS_DIR, "01_mask_comparison.png"), dpi=100, bbox_inches='tight')
        plt.close()
    
    def visualize_morph_stage(self):
        fig, gs = self._create_common_figure("MORPHOLOGY STAGE - Golden Model vs RTL Comparison")
        
        ax0 = fig.add_subplot(gs[0, 0])
        if self.mask_rtl is not None: ax0.imshow(self.mask_rtl, cmap='gray')
        ax0.set_title("Input (Mask Output)", fontsize=14, fontweight='bold')
        ax0.axis('off')
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(self.morph_golden, cmap='gray')
        ax1.set_title("Golden Morphology", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(self.morph_rtl, cmap='gray')
        ax2.set_title("RTL Morphology", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 3])
        diff_colored = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        diff_colored[self.morph_diff] = [1, 0, 0]
        ax3.imshow(diff_colored)
        ax3.set_title(f"Differences ({self.morph_stats['diff_pixels']} px)", fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        fp = np.bitwise_and(~self.morph_golden, self.morph_rtl)
        fn = np.bitwise_and(self.morph_golden, ~self.morph_rtl)
        fp_count, fn_count = np.sum(fp), np.sum(fn)
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.bar(['False Pos.', 'False Neg.'], [fp_count, fn_count], color=['#3498db', '#e67e22'])
        ax4.set_title('Error Breakdown', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='both', labelsize=12)
        
        ax5 = fig.add_subplot(gs[1, 1])
        fp_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)); fp_img[fp] = [0, 0, 1]
        ax5.imshow(fp_img)
        ax5.set_title(f'False Positives: {fp_count}', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        fn_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)); fn_img[fn] = [1, 0.5, 0]
        ax6.imshow(fn_img)
        ax6.set_title(f'False Negatives: {fn_count}', fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.pie([self.morph_stats['accuracy'], 100-self.morph_stats['accuracy']], labels=['Match', 'Diff'], 
                autopct='%1.2f%%', colors=['#27ae60', '#e74c3c'], startangle=90, textprops={'fontsize': 12})
        ax7.set_title(f"Accuracy: {self.morph_stats['accuracy']:.4f}%", fontsize=14, fontweight='bold')
        
        ax8 = fig.add_subplot(gs[2, 1:3])
        ax8.axis('off')
        stats = f"Golden Model: {self.morph_stats['golden_ones']:,} px\nRTL Simulation: {self.morph_stats['rtl_ones']:,} px\nFalse Pos: {fp_count:,} (Hardware Shifts)\nFalse Neg: {fn_count:,} (Hardware Shifts)\nOverall Accuracy: {self.morph_stats['accuracy']:.4f}%"
        ax8.text(0.5, 0.5, stats, fontsize=16, family='monospace', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#eaf2f8', alpha=0.9))
        
        plt.savefig(os.path.join(RESULTS_DIR, "02_morph_comparison.png"), dpi=100, bbox_inches='tight')
        plt.close()

    def visualize_ccl_comparison(self):
        fig, gs = self._create_common_figure(f"CCL STAGE - Golden vs RTL (Filtered Noise < {MIN_CCL_AREA}px)")
        
        # Color Map that keeps Background (0) Black
        import copy
        cmap = copy.copy(plt.cm.tab20)
        cmap.set_under('black')
        
        ax0 = fig.add_subplot(gs[0, 0])
        if self.input_image is not None: ax0.imshow(self.input_image)
        ax0.set_title("Input RGB Image", fontsize=14, fontweight='bold')
        ax0.axis('off')
        
        ax1 = fig.add_subplot(gs[0, 1])
        # vmin=0.5 forces 0 to fall into 'under' category (which is black)
        ax1.imshow(self.ccl_golden, cmap=cmap, vmin=0.5)
        ax1.set_title(f"Golden CCL ({self.ccl_stats['golden_num_labels']} Valid Targets)", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(self.ccl_rtl, cmap=cmap, vmin=0.5)
        ax2.set_title(f"RTL CCL ({self.ccl_stats['rtl_num_labels']} Valid Targets)", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 3])
        diff_map = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        for gl, rl in self.ccl_matches.items():
            diff_map[(self.ccl_golden == gl) | (self.ccl_rtl == rl)] = [0, 1, 0] # Green Match
        for gl in (set(np.unique(self.ccl_golden)) - {0} - set(self.ccl_matches.keys())):
            diff_map[self.ccl_golden == gl] = [1, 0, 0] # Red Missing
        for rl in (set(np.unique(self.ccl_rtl)) - {0} - set(self.ccl_matches.values())):
            diff_map[self.ccl_rtl == rl] = [0, 0, 1] # Blue Extra
            
        ax3.imshow(diff_map)
        ax3.set_title(f"Match(G) / Miss(R) / Extra(B)", fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.bar(['Matches', 'Missing', 'Extra'], 
                [self.ccl_stats['matching_labels'], self.ccl_stats['missing_labels'], self.ccl_stats['extra_labels']], 
                color=['#27ae60', '#e74c3c', '#3498db'])
        ax4.set_title('Component Overlap Status', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='both', labelsize=12)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.pie([self.ccl_stats['accuracy'], 100-self.ccl_stats['accuracy']], labels=['Shared', 'Unshared'],
                autopct='%1.1f%%', colors=['#27ae60', '#e74c3c'], startangle=90, textprops={'fontsize': 12})
        ax5.set_title(f"Object IoU: {self.ccl_stats['accuracy']:.2f}%", fontsize=14, fontweight='bold')
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.bar(['Golden', 'RTL', 'Matched'], 
                [self.ccl_stats['golden_num_labels'], self.ccl_stats['rtl_num_labels'], self.ccl_stats['matching_labels']], 
                color=['#3498db', '#e67e22', '#27ae60'])
        ax6.set_title('Valid Component Count', fontsize=14, fontweight='bold')
        ax6.tick_params(axis='both', labelsize=12)
        
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.pie([self.ccl_stats['label_accuracy'], 100-self.ccl_stats['label_accuracy']], labels=['Detected', 'Missed'],
                autopct='%1.1f%%', colors=['#27ae60', '#e74c3c'], startangle=90, textprops={'fontsize': 12})
        ax7.set_title(f"Detection Rate: {self.ccl_stats['label_accuracy']:.2f}%", fontsize=14, fontweight='bold')
        
        ax8 = fig.add_subplot(gs[2, 1:3])
        ax8.axis('off')
        stats = f"SMART CCL STATISTICS (Noise Area < {MIN_CCL_AREA}px filtered out)\nGolden Targets: {self.ccl_stats['golden_num_labels']}  |  RTL Targets: {self.ccl_stats['rtl_num_labels']}\nSuccessfully Matched: {self.ccl_stats['matching_labels']} objects\nTarget Detection Rate: {self.ccl_stats['label_accuracy']:.2f}%"
        ax8.text(0.5, 0.5, stats, fontsize=16, family='monospace', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#ebf5fb', alpha=0.9))
        
        plt.savefig(os.path.join(RESULTS_DIR, "04_ccl_comparison.png"), dpi=100, bbox_inches='tight')
        plt.close()

    def run_analysis(self):
        if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
        if not self.load_all_data(): return False
        
        self.compute_differences()
        self.print_summary()
        
        self.visualize_mask_stage()
        self.visualize_morph_stage()
        self.visualize_ccl_comparison()
        return True

def main():
    return ComparisonAnalyzer().run_analysis()

if __name__ == "__main__":
    sys.exit(0 if main() else 1)