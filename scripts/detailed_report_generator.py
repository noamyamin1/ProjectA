"""
Detailed Report Generator
Comprehensive analysis and detailed statistics for verification
Generates HTML and text reports with detailed metrics
"""

import os
import sys
import numpy as np
from datetime import datetime
import json

# Paths
BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Image parameters
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Reference files
MASK_OUT_GOLDEN = os.path.join(DATA_DIR, "mask_out2.txt")
MORPH_OUT_GOLDEN = os.path.join(DATA_DIR, "morph_out2.txt")
MASK_OUT_RTL = os.path.join(DATA_DIR, "actual_mask_out.txt")
MORPH_OUT_RTL = os.path.join(DATA_DIR, "actual_morph_out.txt")


class ReportGenerator:
    """Generates comprehensive analysis reports"""
    
    def __init__(self):
        """Initialize report generator"""
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_data = {
            'timestamp': self.timestamp,
            'stage_results': {}
        }
    
    def load_binary_file(self, filepath, num_pixels):
        """Load binary data from file"""
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
    
    def analyze_stage(self, stage_name, golden_data, rtl_data):
        """Analyze a processing stage"""
        num_pixels = IMG_HEIGHT * IMG_WIDTH
        
        golden_ones = np.sum(golden_data)
        rtl_ones = np.sum(rtl_data)
        
        # Calculate differences
        xor_diff = np.bitwise_xor(golden_data, rtl_data)
        diff_pixels = np.sum(xor_diff)
        
        false_positive = np.bitwise_and(~golden_data, rtl_data)
        false_negative = np.bitwise_and(golden_data, ~rtl_data)
        
        fp_count = np.sum(false_positive)
        fn_count = np.sum(false_negative)
        true_positive = np.sum(np.bitwise_and(golden_data, rtl_data))
        true_negative = np.sum(np.bitwise_and(~golden_data, ~rtl_data))
        
        accuracy = 100.0 * (true_positive + true_negative) / num_pixels
        
        # Precision and Recall
        precision = 100.0 * true_positive / (true_positive + fp_count) if (true_positive + fp_count) > 0 else 0
        recall = 100.0 * true_positive / (true_positive + fn_count) if (true_positive + fn_count) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Spatial analysis
        diff_regions = self.find_error_regions(xor_diff)
        
        results = {
            'stage': stage_name,
            'total_pixels': num_pixels,
            'golden_ones': int(golden_ones),
            'golden_percent': 100.0 * golden_ones / num_pixels,
            'rtl_ones': int(rtl_ones),
            'rtl_percent': 100.0 * rtl_ones / num_pixels,
            'true_positive': int(true_positive),
            'true_negative': int(true_negative),
            'false_positive': int(fp_count),
            'false_negative': int(fn_count),
            'total_errors': int(diff_pixels),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'error_regions': diff_regions,
            'file_golden': os.path.basename(self.get_stage_file_golden(stage_name)),
            'file_rtl': os.path.basename(self.get_stage_file_rtl(stage_name))
        }
        
        return results
    
    def get_stage_file_golden(self, stage_name):
        """Get golden reference file for stage"""
        if stage_name == "Red Mask":
            return MASK_OUT_GOLDEN
        elif stage_name == "Morphology":
            return MORPH_OUT_GOLDEN
    
    def get_stage_file_rtl(self, stage_name):
        """Get RTL output file for stage"""
        if stage_name == "Red Mask":
            return MASK_OUT_RTL
        elif stage_name == "Morphology":
            return MORPH_OUT_RTL
    
    def find_error_regions(self, diff_mask, region_size=64):
        """Find regions with errors"""
        regions = []
        
        if diff_mask is None or np.sum(diff_mask) == 0:
            return regions
        
        # Divide image into regions and count errors
        for y in range(0, IMG_HEIGHT, region_size):
            for x in range(0, IMG_WIDTH, region_size):
                y_end = min(y + region_size, IMG_HEIGHT)
                x_end = min(x + region_size, IMG_WIDTH)
                
                region_errors = np.sum(diff_mask[y:y_end, x:x_end])
                if region_errors > 0:
                    regions.append({
                        'y_start': y,
                        'y_end': y_end,
                        'x_start': x,
                        'x_end': x_end,
                        'error_count': int(region_errors)
                    })
        
        # Sort by error count (most errors first)
        regions.sort(key=lambda r: r['error_count'], reverse=True)
        return regions[:10]  # Top 10 error regions
    
    def generate_text_report(self):
        """Generate text-based report"""
        # Load data
        mask_golden = self.load_binary_file(MASK_OUT_GOLDEN, IMG_HEIGHT * IMG_WIDTH)
        mask_rtl = self.load_binary_file(MASK_OUT_RTL, IMG_HEIGHT * IMG_WIDTH)
        morph_golden = self.load_binary_file(MORPH_OUT_GOLDEN, IMG_HEIGHT * IMG_WIDTH)
        morph_rtl = self.load_binary_file(MORPH_OUT_RTL, IMG_HEIGHT * IMG_WIDTH)
        
        # Analyze stages
        print("\nAnalyzing Red Mask Stage...")
        mask_results = self.analyze_stage("Red Mask", mask_golden, mask_rtl)
        self.report_data['stage_results']['Red Mask'] = mask_results
        
        print("Analyzing Morphology Stage...")
        morph_results = self.analyze_stage("Morphology", morph_golden, morph_rtl)
        self.report_data['stage_results']['Morphology'] = morph_results
        
        # Generate text report
        report_file = os.path.join(RESULTS_DIR, "detailed_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ROAD SIGN DETECTOR - VERIFICATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {self.timestamp}\n")
            f.write(f"Image Resolution: {IMG_WIDTH}x{IMG_HEIGHT}\n")
            f.write(f"Total Pixels: {IMG_HEIGHT * IMG_WIDTH:,}\n")
            f.write("="*80 + "\n\n")
            
            # Report each stage
            for stage_name, results in self.report_data['stage_results'].items():
                self.write_stage_report(f, stage_name, results)
            
            # Summary section
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            total_accuracy = np.mean([r['accuracy'] for r in self.report_data['stage_results'].values()])
            total_f1 = np.mean([r['f1_score'] for r in self.report_data['stage_results'].values()])
            
            f.write(f"Average Accuracy: {total_accuracy:.6f}%\n")
            f.write(f"Average F1-Score: {total_f1:.6f}\n\n")
            
            # Status
            status = "PASS" if total_accuracy >= 99.99 else "FAIL"
            f.write(f"Verification Status: {status}\n")
            f.write("\n" + "="*80 + "\n")
        
        print(f"\n✓ Text report saved: {report_file}")
        return report_file
    
    def write_stage_report(self, f, stage_name, results):
        """Write report section for a stage"""
        f.write(f"\n{stage_name.upper()} STAGE\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Source Files:\n")
        f.write(f"  Golden: {results['file_golden']}\n")
        f.write(f"  RTL:    {results['file_rtl']}\n\n")
        
        f.write(f"Pixel Counts:\n")
        f.write(f"  Total Pixels:        {results['total_pixels']:>10,}\n")
        f.write(f"  Golden Match Pixels: {results['golden_ones']:>10,} "
               f"({results['golden_percent']:>6.2f}%)\n")
        f.write(f"  RTL Match Pixels:    {results['rtl_ones']:>10,} "
               f"({results['rtl_percent']:>6.2f}%)\n\n")
        
        f.write(f"Confusion Matrix:\n")
        f.write(f"  True Positive:       {results['true_positive']:>10,}\n")
        f.write(f"  True Negative:       {results['true_negative']:>10,}\n")
        f.write(f"  False Positive:      {results['false_positive']:>10,}\n")
        f.write(f"  False Negative:      {results['false_negative']:>10,}\n\n")
        
        f.write(f"Performance Metrics:\n")
        f.write(f"  Accuracy:            {results['accuracy']:>10.6f}%\n")
        f.write(f"  Precision:           {results['precision']:>10.6f}%\n")
        f.write(f"  Recall:              {results['recall']:>10.6f}%\n")
        f.write(f"  F1-Score:            {results['f1_score']:>10.6f}\n")
        f.write(f"  Total Errors:        {results['total_errors']:>10,}\n\n")
        
        if results['error_regions']:
            f.write(f"Top Error Regions (by pixel count):\n")
            for i, region in enumerate(results['error_regions'][:5], 1):
                f.write(f"  {i}. Region [{region['y_start']}:{region['y_end']}, "
                       f"{region['x_start']}:{region['x_end']}] - "
                       f"{region['error_count']} errors\n")
        else:
            f.write("No error regions found (Perfect match!)\n")
        
        f.write("\n")
    
    def generate_json_report(self):
        """Generate JSON report for programmatic access"""
        # Load data if not already done
        if not self.report_data['stage_results']:
            mask_golden = self.load_binary_file(MASK_OUT_GOLDEN, IMG_HEIGHT * IMG_WIDTH)
            mask_rtl = self.load_binary_file(MASK_OUT_RTL, IMG_HEIGHT * IMG_WIDTH)
            morph_golden = self.load_binary_file(MORPH_OUT_GOLDEN, IMG_HEIGHT * IMG_WIDTH)
            morph_rtl = self.load_binary_file(MORPH_OUT_RTL, IMG_HEIGHT * IMG_WIDTH)
            
            self.analyze_stage("Red Mask", mask_golden, mask_rtl)
            self.analyze_stage("Morphology", morph_golden, morph_rtl)
        
        json_file = os.path.join(RESULTS_DIR, "verification_report.json")
        
        with open(json_file, 'w') as f:
            json.dump(self.report_data, f, indent=2)
        
        print(f"✓ JSON report saved: {json_file}")
        return json_file
    
    def run(self):
        """Run report generation"""
        print("\n" + "="*80)
        print("DETAILED REPORT GENERATOR")
        print("="*80)
        
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        # Generate both report types
        text_report = self.generate_text_report()
        json_report = self.generate_json_report()
        
        # Print summary to console
        print("\n" + "="*80)
        print("REPORT SUMMARY")
        print("="*80)
        
        for stage_name, results in self.report_data['stage_results'].items():
            print(f"\n[{stage_name}]")
            print(f"  Accuracy: {results['accuracy']:.6f}%")
            print(f"  Errors:   {results['total_errors']} pixels")
            print(f"  F1-Score: {results['f1_score']:.4f}")
        
        print("\n✓ All reports generated successfully!")
        print(f"  Text Report: {text_report}")
        print(f"  JSON Report: {json_report}")
        print("="*80 + "\n")


def main():
    """Main entry point"""
    generator = ReportGenerator()
    generator.run()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
