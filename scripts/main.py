#!/usr/bin/env python3
"""
Golden Model vs Testbench Comparison - Main Entry Point
========================================================

This is the primary workflow for comparing RTL testbench outputs 
against the golden reference model.

Usage:
    python main.py --help           # Show available options
    python main.py                  # Full verification workflow
    python main.py --golden-only    # Generate golden model only
    python main.py --compare-only   # Run comparison only (no golden gen)
    python main.py --report-only    # Generate reports only
"""

import os
import sys
import argparse
from datetime import datetime

# Configuration
BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def run_script(script_name, description):
    """Run a Python script with error handling"""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    
    if not os.path.exists(script_path):
        print(f"\n✗ ERROR: Script not found: {script_name}")
        return False
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running: {description}")
    print(f"  Script: {script_name}")
    
    try:
        import subprocess
        result = subprocess.run(
            ['python3', script_path],
            cwd=SCRIPTS_DIR,
            check=True
        )
        print(f"✓ Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {description}")
        print(f"  Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False

def main():
    """Main workflow orchestrator"""
    parser = argparse.ArgumentParser(
        description="Golden Model vs Testbench Comparison Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Full verification pipeline
  python main.py --golden-only    # Generate golden model only
  python main.py --compare-only   # Run comparison (no golden gen)
  python main.py --report-only    # Generate reports only
  python main.py --setup          # Validate environment only
        """
    )
    
    parser.add_argument(
        '--golden-only',
        action='store_true',
        help='Generate golden model only (skip comparisons)'
    )
    parser.add_argument(
        '--compare-only',
        action='store_true',
        help='Run comparisons only (assume golden already exists)'
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate reports only'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Validate environment and exit'
    )
    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='Generate visualizations only'
    )
    
    args = parser.parse_args()
    
    # Print intro
    print_header("GOLDEN MODEL vs TESTBENCH COMPARISON")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scripts Dir: {SCRIPTS_DIR}")
    print(f"Base Dir: {BASE_DIR}")
    
    # Validation step (always runs)
    print_header("Step 0: Environment Validation")
    if not run_script('setup_and_doctor.py', 'Environment validation'):
        print("\n✗ Environment validation failed. Please fix issues above.")
        return 1
    
    if args.setup:
        print("\n✓ Setup validation complete (--setup mode)")
        return 0
    
    # Golden model generation
    if not args.compare_only and not args.report_only and not args.visualize_only:
        print_header("Step 1: Golden Model Generation")
        if not run_script('golden_model_complete.py', 'Generate golden reference model'):
            print("\n✗ Golden model generation failed.")
            return 1
            
    if args.golden_only:
        print("\n✓ Golden model generation complete (--golden-only mode)")
        return 0

    # Step 2: CCL Resolved Reconstruction
    print_header("Step 2: CCL Resolved Reconstruction")
    print("Attempting to reconstruct fully resolved CCL labels from RTL pass1 + parent table...")
    if not run_script('ccl_resolved_reconstructor.py', 'Reconstruct CCL resolved labels'):
        print("\n⚠ CCL reconstruction skipped or failed")

    # Step 3: Comparison and analysis (Now includes Smart CCL Visualization)
    if not args.report_only and not args.visualize_only:
        print_header("Step 3: Detailed Comparison & Analysis")
        
        comparison_success = True
        if run_script('comparison_and_analysis.py', 'Comprehensive golden vs RTL analysis'):
            print("  ✓ comparison_and_analysis.py completed")
        else:
            print("  ✗ comparison_and_analysis.py failed")
            comparison_success = False
            
        if run_script('end_to_end_comparison.py', 'Stage-by-stage comparison'):
            print("  ✓ end_to_end_comparison.py completed")
        else:
            print("  ✗ end_to_end_comparison.py failed")
            
        if not comparison_success and args.compare_only:
            return 1

    # Step 4: Report generation
    if not args.visualize_only:
        print_header("Step 4: Report Generation")
        if not run_script('detailed_report_generator.py', 'Generate detailed reports'):
            print("\n✗ Report generation failed.")
            return 1

    # Step 5: Final Template Matching Validation
    if not args.report_only:
        print_header("Step 5: Template Matching Validation")
        if run_script('template_matching_analyzer.py', 'Template Matching Validation (HW Logic)'):
            print("  ✓ Template Matching visualizations generated")
        else:
            print("  ⚠ Template Matching generation had issues")

    # Final summary
    print_header("VERIFICATION COMPLETE")
    print("\nGenerated Outputs:")
    print(f"  • Detailed reports: {os.path.join(BASE_DIR, 'results')}/")
    print(f"  • Visualizations: {os.path.join(BASE_DIR, 'results')}/")
    print("\nEnd Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())