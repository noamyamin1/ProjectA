#!/usr/bin/env python3
"""
Unified Verification & Analysis Tool
Orchestrates all verification steps: golden model generation, comparison, and reporting
This is the main entry point for complete verification workflow
"""

import os
import sys
import subprocess
import time
from datetime import datetime

BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")


class VerificationOrchestrator:
    """Orchestrates complete verification workflow"""
    
    def __init__(self):
        """Initialize orchestrator"""
        self.start_time = None
        self.results = {}
        self.create_output_dirs()
    
    def create_output_dirs(self):
        """Create necessary output directories"""
        for dir_path in [RESULTS_DIR, DATA_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"✓ Created directory: {dir_path}")
    
    def run_script(self, script_name, description):
        """Run a Python script and track results"""
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        
        if not os.path.exists(script_path):
            print(f"✗ Script not found: {script_path}")
            return False
        
        print(f"\n{'='*70}")
        print(f"▶ {description}")
        print(f"{'='*70}")
        print(f"Running: {script_name}")
        
        try:
            start = time.time()
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=SCRIPTS_DIR,
                capture_output=False,
                timeout=300  # 5 minute timeout
            )
            elapsed = time.time() - start
            
            success = result.returncode == 0
            self.results[script_name] = {
                'success': success,
                'elapsed': elapsed,
                'description': description
            }
            
            if success:
                print(f"✓ {description} completed in {elapsed:.2f}s")
            else:
                print(f"✗ {description} failed with code {result.returncode}")
            
            return success
        except subprocess.TimeoutExpired:
            print(f"✗ {description} timed out")
            self.results[script_name] = {
                'success': False,
                'elapsed': 300,
                'description': description
            }
            return False
        except Exception as e:
            print(f"✗ Error running {description}: {e}")
            self.results[script_name] = {
                'success': False,
                'elapsed': 0,
                'description': description
            }
            return False
    
    def print_verification_banner(self):
        """Print welcome banner"""
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " ROAD SIGN DETECTOR - VERIFICATION & ANALYSIS SUITE ".center(68) + "║")
        print("║" + " Golden Model + Comparison + Reporting ".center(68) + "║")
        print("╚" + "="*68 + "╝")
        print()
    
    def print_summary(self):
        """Print execution summary"""
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " VERIFICATION WORKFLOW SUMMARY ".center(68) + "║")
        print("╚" + "="*68 + "╝")
        
        total_time = sum(r['elapsed'] for r in self.results.values())
        successful = sum(1 for r in self.results.values() if r['success'])
        total = len(self.results)
        
        print(f"\nExecution Summary:")
        print(f"  Total Steps:     {total}")
        print(f"  Successful:      {successful}/{total}")
        print(f"  Total Time:      {total_time:.2f} seconds")
        
        print(f"\nStep Details:")
        for script, result in self.results.items():
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            print(f"  [{status}] {result['description']} ({result['elapsed']:.2f}s)")
        
        print(f"\nOutput Directories:")
        print(f"  Data:    {DATA_DIR}")
        print(f"  Results: {RESULTS_DIR}")
        
        print("\nGenerated Files:")
        self.print_generated_files()
        
        if successful == total:
            print("\n" + "="*70)
            print("✓ ALL VERIFICATION STEPS COMPLETED SUCCESSFULLY!")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("✗ Some verification steps failed. Check output above.")
            print("="*70)
        
        print()
    
    def print_generated_files(self):
        """Print list of generated files"""
        # Data files
        data_files = []
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                if f.startswith(('mask_out', 'morph_out', 'ccl_labels', 'geom_bboxes', 'image_in')):
                    size = os.path.getsize(os.path.join(DATA_DIR, f))
                    data_files.append(f"  - {f} ({size:,} bytes)")
        
        # Result files
        result_files = []
        if os.path.exists(RESULTS_DIR):
            for f in os.listdir(RESULTS_DIR):
                path = os.path.join(RESULTS_DIR, f)
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    result_files.append(f"  - {f} ({size:,} bytes)")
        
        if data_files:
            print(f"\n  Reference Data (data/):")
            for f in sorted(data_files)[:10]:
                print(f)
            if len(data_files) > 10:
                print(f"  ... and {len(data_files)-10} more")
        
        if result_files:
            print(f"\n  Analysis Results (results/):")
            for f in sorted(result_files)[:10]:
                print(f)
            if len(result_files) > 10:
                print(f"  ... and {len(result_files)-10} more")
    
    def run_complete_verification(self):
        """Run complete verification workflow"""
        self.print_verification_banner()
        self.start_time = datetime.now()
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Step 1: Generate golden model
        step1_ok = self.run_script(
            'golden_model_complete.py',
            'Step 1: Generate Golden Model Reference'
        )
        
        if not step1_ok:
            print("\n✗ Failed to generate golden model. Stopping verification.")
            return False
        
        # Step 2: Compare RTL vs Golden
        step2_ok = self.run_script(
            'comparison_and_analysis.py',
            'Step 2: Compare RTL Simulation vs Golden Model'
        )
        
        # Step 3: Generate detailed reports
        step3_ok = self.run_script(
            'detailed_report_generator.py',
            'Step 3: Generate Detailed Analysis Reports'
        )
        
        # Print summary
        self.print_summary()
        
        return step1_ok and step2_ok and step3_ok
    
    def print_quick_reference(self):
        """Print quick reference guide"""
        print("\n" + "="*70)
        print("QUICK REFERENCE - HOW TO USE")
        print("="*70)
        
        print("""
To run complete verification workflow:
  $ cd /users/epnyrk/Project/design/work/ProjectA/scripts
  $ python3 run_verification.py

Individual Scripts (advanced):
  1. Generate Golden Model:
     $ python3 golden_model_complete.py
  
  2. Compare & Analyze:
     $ python3 comparison_and_analysis.py
  
  3. Generate Reports:
     $ python3 detailed_report_generator.py

Output Files:
  • Reference data:  /users/epnyrk/Project/design/work/ProjectA/data/
  • Analysis plots:  /users/epnyrk/Project/design/work/ProjectA/results/
  • Reports:         /users/epnyrk/Project/design/work/ProjectA/results/

Key Output Files:
  - golden_model_complete.py  → Generates reference outputs
  - comparison_and_analysis.py → Creates visualization PNGs
  - detailed_report_generator.py → Creates detailed reports

Generated Visualization Files:
  - 01_mask_comparison.png    (Red mask accuracy)
  - 02_morph_comparison.png   (Morphology filter accuracy)
  - 03_pipeline_flow.png      (Overall pipeline flow)
  - detailed_report.txt       (Text-based report)
  - verification_report.json  (Machine-readable results)
""")


def main():
    """Main entry point"""
    # Check if running with --help flag
    if '--help' in sys.argv or '-h' in sys.argv:
        orchestrator = VerificationOrchestrator()
        orchestrator.print_quick_reference()
        return True
    
    # Run complete verification
    orchestrator = VerificationOrchestrator()
    success = orchestrator.run_complete_verification()
    
    # Print quick reference at the end
    orchestrator.print_quick_reference()
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n✗ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
