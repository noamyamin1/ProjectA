"""
Golden Model for Road Sign Detection - Complete Image Processing Pipeline
Implements: Red Mask -> Morphology -> CCL -> Template Matching
Output: Reference data for comparison with RTL simulation
"""

import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label as scipy_label
import struct

# Configuration
BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "pyton/pics_to_test/slippery_road_redcar.jpg")
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATE_DIR = os.path.join(BASE_DIR, "pyton/Templates")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Image parameters
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# Algorithm parameters
MIN_RED_VAL = 15
MARGIN_SHIFT = 3
MORPH_KERNEL = 3  # 3x3 kernel for dilation/erosion

# Output files
IMAGE_IN_HEX = os.path.join(DATA_DIR, "image_in2.hex")
MASK_OUT_REF = os.path.join(DATA_DIR, "mask_out2.txt")
MORPH_OUT_REF = os.path.join(DATA_DIR, "morph_out2.txt")
CCL_LABELS_REF = os.path.join(DATA_DIR, "ccl_labels_golden.txt")
GEOM_BBOXES_REF = os.path.join(DATA_DIR, "geom_bboxes_golden.txt")

class GoldenModelPipeline:
    """Complete software implementation of road sign detector pipeline"""
    
    def __init__(self, img_path):
        """Initialize pipeline with input image"""
        self.img_path = img_path
        self.img_np = None
        self.red_mask = None
        self.morph_mask = None
        self.ccl_labels = None
        self.bboxes = None
        self.template_ids = None
        
        # Intermediate results for debugging
        self.r = None
        self.g = None
        self.b = None
        
    def load_and_prepare_image(self):
        """Load image and prepare for processing"""
        if not os.path.exists(self.img_path):
            print(f"Error: Input file '{self.img_path}' not found.")
            return False
            
        try:
            img = Image.open(self.img_path).convert('RGB')
            
            if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
                print(f"Info: Resizing from {img.size} to FHD {TARGET_WIDTH}x{TARGET_HEIGHT}")
                img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
            
            self.img_np = np.array(img, dtype=np.uint8)
            self.r = self.img_np[:, :, 0].astype(np.int16)
            self.g = self.img_np[:, :, 1].astype(np.int16)
            self.b = self.img_np[:, :, 2].astype(np.int16)
            
            print(f"✓ Loaded image: {self.img_path}")
            print(f"  Shape: {self.img_np.shape}")
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def stage1_red_mask(self):
        """
        Stage 1: Red Mask Detection
        Implementation of red detection algorithm matching RTL behavior
        """
        print("\n[Stage 1] Computing Red Mask Detection...")
        
        # Calculate dynamic margin based on red channel
        margin = self.r >> MARGIN_SHIFT  # Right shift by 3 = divide by 8
        
        # Four conditions that must all be true
        cond1_dom_r_vs_g = (self.r > (self.g + margin))
        cond2_dom_r_vs_b = (self.r > (self.b + margin))
        cond3_orange_killer = (self.g < (self.b + margin))
        cond4_min_red = (self.r > MIN_RED_VAL)
        
        # Combine all conditions with AND logic
        self.red_mask = np.logical_and.reduce((cond1_dom_r_vs_g, cond2_dom_r_vs_b, 
                                               cond3_orange_killer, cond4_min_red))
        
        # Count statistics
        red_pixels = np.sum(self.red_mask)
        total_pixels = TARGET_HEIGHT * TARGET_WIDTH
        red_percent = 100.0 * red_pixels / total_pixels
        
        print(f"  ✓ Red pixels: {red_pixels} ({red_percent:.2f}%)")
        return True
    
    def stage2_morphology(self):
        """
        Stage 2: Morphology Filter (Dilation -> Erosion)
        Implementation: Dilation with 3x3 kernel, then erosion with 3x3 kernel
        """
        print("\n[Stage 2] Computing Morphology Filter...")
        
        # Create 3x3 structuring element for morphological operations
        kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), dtype=bool)
        
        # Dilation: expand regions
        dilated = binary_dilation(self.red_mask, structure=kernel)
        
        # Erosion: shrink regions (closing operation)
        self.morph_mask = binary_erosion(dilated, structure=kernel)
        
        # Statistics
        morph_pixels = np.sum(self.morph_mask)
        red_pixels = np.sum(self.red_mask)
        
        print(f"  ✓ After dilation: {np.sum(dilated)} pixels")
        print(f"  ✓ After morphology: {morph_pixels} pixels")
        print(f"  ✓ Change from red mask: {morph_pixels - red_pixels:+d} pixels")
        return True
    
    def stage3_connected_component_labeling(self):
        """
        Stage 3: Connected Component Labeling (CCL)
        Identifies individual objects from the morphology output
        """
        print("\n[Stage 3] Computing Connected Component Labeling...")
        
        # Use scipy's label function with 8-connectivity
        self.ccl_labels, num_labels = scipy_label(self.morph_mask, structure=np.ones((3,3)))
        
        print(f"  ✓ Found {num_labels} connected components")
        
        # Get statistics for each connected component
        if num_labels > 0:
            component_sizes = np.bincount(self.ccl_labels.ravel())
            print(f"  ✓ Component sizes: min={np.min(component_sizes[1:])}, "
                  f"max={np.max(component_sizes[1:])}, "
                  f"mean={np.mean(component_sizes[1:]):.1f}")
        
        return True
    
    def stage4_geometry_filtering(self):
        """
        Stage 4: Geometry Filtering
        Computes bounding boxes for each connected component
        """
        print("\n[Stage 4] Computing Geometry Filtering...")
        
        bboxes_list = []
        template_ids_list = []
        
        if self.ccl_labels is not None and np.max(self.ccl_labels) > 0:
            num_labels = np.max(self.ccl_labels)
            
            for label_id in range(1, num_labels + 1):
                # Find all pixels belonging to this label
                component_mask = (self.ccl_labels == label_id)
                
                if not np.any(component_mask):
                    continue
                
                # Find bounding box
                rows, cols = np.where(component_mask)
                ymin, ymax = int(np.min(rows)), int(np.max(rows))
                xmin, xmax = int(np.min(cols)), int(np.max(cols))
                
                # Store bounding box
                bboxes_list.append({
                    'label_id': label_id,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'width': xmax - xmin + 1,
                    'height': ymax - ymin + 1,
                    'area': np.sum(component_mask)
                })
                
                # Placeholder: template_id will be populated by matching stage
                template_ids_list.append(0)
            
            self.bboxes = bboxes_list
            self.template_ids = template_ids_list
            
            print(f"  ✓ Extracted {len(self.bboxes)} bounding boxes")
            if self.bboxes:
                areas = [bb['area'] for bb in self.bboxes]
                print(f"  ✓ Area statistics: min={np.min(areas)}, max={np.max(areas)}, "
                      f"mean={np.mean(areas):.1f}")
        else:
            print(f"  ✗ No components to filter")
            self.bboxes = []
            self.template_ids = []
        
        return True
    
    def run_complete_pipeline(self):
        """Execute the complete processing pipeline"""
        print("\n" + "="*70)
        print("GOLDEN MODEL - COMPLETE PROCESSING PIPELINE")
        print("="*70)
        
        if not self.load_and_prepare_image():
            return False
        
        if not self.stage1_red_mask():
            return False
            
        if not self.stage2_morphology():
            return False
            
        if not self.stage3_connected_component_labeling():
            return False
            
        if not self.stage4_geometry_filtering():
            return False
        
        print("\n" + "="*70)
        print("✓ PIPELINE EXECUTION COMPLETE")
        print("="*70)
        return True
    
    def save_outputs(self):
        """Save golden reference outputs"""
        print("\n" + "="*70)
        print("SAVING GOLDEN REFERENCE OUTPUTS")
        print("="*70)
        
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # 1. Save image_in2.hex (RGB input)
        print(f"\nGenerating: {IMAGE_IN_HEX}")
        with open(IMAGE_IN_HEX, 'w') as f:
            for y in range(TARGET_HEIGHT):
                for x in range(TARGET_WIDTH):
                    r_val = self.img_np[y, x, 0]
                    g_val = self.img_np[y, x, 1]
                    b_val = self.img_np[y, x, 2]
                    hex_str = f"00{r_val:02x}{g_val:02x}{b_val:02x}\n"
                    f.write(hex_str)
        print(f"  ✓ Saved {TARGET_HEIGHT * TARGET_WIDTH} pixels")
        
        # 2. Save mask_out2.txt (Red mask bits)
        print(f"\nGenerating: {MASK_OUT_REF}")
        with open(MASK_OUT_REF, 'w') as f:
            for y in range(TARGET_HEIGHT):
                for x in range(TARGET_WIDTH):
                    bit = '1' if self.red_mask[y, x] else '0'
                    f.write(bit + '\n')
        red_count = np.sum(self.red_mask)
        print(f"  ✓ Saved {red_count} red pixels out of {TARGET_HEIGHT * TARGET_WIDTH}")
        
        # 3. Save morph_out2.txt (Morphology output bits)
        print(f"\nGenerating: {MORPH_OUT_REF}")
        with open(MORPH_OUT_REF, 'w') as f:
            for y in range(TARGET_HEIGHT):
                for x in range(TARGET_WIDTH):
                    bit = '1' if self.morph_mask[y, x] else '0'
                    f.write(bit + '\n')
        morph_count = np.sum(self.morph_mask)
        print(f"  ✓ Saved {morph_count} morphology pixels")
        
        # 4. Save CCL labels
        print(f"\nGenerating: {CCL_LABELS_REF}")
        with open(CCL_LABELS_REF, 'w') as f:
            for y in range(TARGET_HEIGHT):
                for x in range(TARGET_WIDTH):
                    label_val = self.ccl_labels[y, x]
                    f.write(f"{label_val} ")
                if (y + 1) % 10 == 0:
                    f.write("\n")
        print(f"  ✓ Saved CCL labels for {TARGET_HEIGHT}x{TARGET_WIDTH} image")
        
        # 5. Save geometry bboxes
        print(f"\nGenerating: {GEOM_BBOXES_REF}")
        with open(GEOM_BBOXES_REF, 'w') as f:
            f.write(f"# Bounding Boxes from Golden Model\n")
            f.write(f"# Format: id xmin xmax ymin ymax width height area\n")
            f.write(f"# Total Components: {len(self.bboxes)}\n\n")
            for i, bb in enumerate(self.bboxes):
                f.write(f"{bb['label_id']} {bb['xmin']} {bb['xmax']} {bb['ymin']} "
                       f"{bb['ymax']} {bb['width']} {bb['height']} {bb['area']}\n")
        print(f"  ✓ Saved {len(self.bboxes)} bounding boxes")
        
        print("\n✓ All golden reference outputs saved successfully")
        return True
    
    def print_summary(self):
        """Print summary of processing results"""
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        print(f"\nImage: {self.img_path}")
        print(f"Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
        print(f"\nStage 1 - Red Mask:")
        print(f"  Red pixels: {np.sum(self.red_mask)}")
        print(f"\nStage 2 - Morphology:")
        print(f"  Morphology pixels: {np.sum(self.morph_mask)}")
        print(f"\nStage 3 - CCL:")
        print(f"  Connected components: {np.max(self.ccl_labels) if self.ccl_labels is not None else 0}")
        print(f"\nStage 4 - Geometry:")
        print(f"  Bounding boxes: {len(self.bboxes)}")
        print("="*70)


def main():
    """Main entry point"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " ROAD SIGN DETECTOR - GOLDEN MODEL REFERENCE GENERATOR ".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    # Create pipeline
    pipeline = GoldenModelPipeline(INPUT_IMAGE_PATH)
    
    # Run complete pipeline
    if not pipeline.run_complete_pipeline():
        print("✗ Pipeline execution failed")
        return False
    
    # Save outputs
    if not pipeline.save_outputs():
        print("✗ Failed to save outputs")
        return False
    
    # Print summary
    pipeline.print_summary()
    
    print("\n✓ Golden model generation complete!")
    print(f"  Reference files saved to: {DATA_DIR}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
