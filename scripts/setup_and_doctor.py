#!/usr/bin/env python3
"""
Setup & Doctor Tool for Road Sign Detector Verification Suite
Checks dependencies, validates file structure, and provides diagnostics
"""

import os
import sys
import subprocess

BASE_DIR = "/users/epnyrk/Project/design/work/ProjectA"
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def check_python_version():
    """Check Python version"""
    print("\n[1] Python Version Check")
    print(f"    Current: Python {sys.version}")
    
    if sys.version_info >= (3, 7):
        print("    ✓ Python version OK")
        return True
    else:
        print("    ✗ Python 3.7+ required")
        return False


def check_dependencies():
    """Check required Python packages"""
    print("\n[2] Python Dependencies Check")
    
    required_packages = {
        'numpy': 'Numerical computing',
        'scipy': 'Scientific computing & filtering',
        'matplotlib': 'Plotting & visualization',
        'PIL': 'Image processing'
    }
    
    missing = []
    
    for package, description in required_packages.items():
        try:
            if package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"    ✓ {package:15} - {description}")
        except ImportError:
            print(f"    ✗ {package:15} - {description} [MISSING]")
            missing.append(package)
    
    if missing:
        print(f"\n    To install missing packages:")
        print(f"    $ pip install {' '.join(missing)}")
        return False
    
    return True


def check_directory_structure():
    """Check if required directories exist"""
    print("\n[3] Directory Structure Check")
    
    directories = {
        'scripts': SCRIPTS_DIR,
        'data': DATA_DIR,
        'results': RESULTS_DIR,
    }
    
    all_ok = True
    
    for name, path in directories.items():
        if os.path.exists(path):
            print(f"    ✓ {name:15} {path}")
        else:
            print(f"    ✗ {name:15} {path} [MISSING]")
            all_ok = False
    
    return all_ok


def check_script_files():
    """Check if required scripts exist"""
    print("\n[4] Script Files Check")
    
    required_scripts = [
        'golden_model_complete.py',
        'comparison_and_analysis.py',
        'detailed_report_generator.py',
        'run_verification.py',
        'setup_and_doctor.py',
        'README.md'
    ]
    
    all_ok = True
    
    for script in required_scripts:
        script_path = os.path.join(SCRIPTS_DIR, script)
        if os.path.exists(script_path):
            size = os.path.getsize(script_path)
            print(f"    ✓ {script:40} ({size:,} bytes)")
        else:
            print(f"    ✗ {script:40} [MISSING]")
            all_ok = False
    
    return all_ok


def check_input_image():
    """Check if input image exists"""
    print("\n[5] Input Image Check")
    
    image_path = os.path.join(BASE_DIR, "pyton/pics_to_test/slippery_road_redcar.jpg")
    
    if os.path.exists(image_path):
        size = os.path.getsize(image_path)
        print(f"    ✓ Input image found")
        print(f"      {image_path}")
        print(f"      Size: {size:,} bytes")
        return True
    else:
        print(f"    ✗ Input image not found")
        print(f"      Expected: {image_path}")
        return False


def check_rtl_outputs():
    """Check if RTL simulation outputs exist"""
    print("\n[6] RTL Simulation Outputs Check")
    
    rtl_files = {
        'actual_mask_out.txt': 'Red mask output from RTL',
        'actual_morph_out.txt': 'Morphology output from RTL',
    }
    
    any_missing = False
    
    for filename, description in rtl_files.items():
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"    ✓ {filename:25} ({size:,} bytes)")
        else:
            print(f"    ✗ {filename:25} [MISSING - Run RTL sim first]")
            any_missing = True
    
    if any_missing:
        print("\n    Note: RTL simulation must be completed before running comparison")
        print("          Golden model generation doesn't require RTL outputs")
    
    return not any_missing


def check_disk_space():
    """Check available disk space"""
    print("\n[7] Disk Space Check")
    
    try:
        import shutil
        stat = shutil.disk_usage(BASE_DIR)
        
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        
        print(f"    Available: {free_gb:.2f} GB / {total_gb:.2f} GB")
        
        if free_gb > 1:
            print(f"    ✓ Sufficient disk space (>1 GB available)")
            return True
        else:
            print(f"    ✗ Low disk space (<1 GB available)")
            return False
    except Exception as e:
        print(f"    ? Could not determine disk space: {e}")
        return True  # Don't fail on this


def install_dependencies():
    """Install missing dependencies"""
    print("\n" + "="*70)
    print("Installing Missing Dependencies")
    print("="*70)
    
    packages = ['numpy', 'scipy', 'matplotlib', 'Pillow']
    
    try:
        import subprocess
        print(f"\nRunning: pip install {' '.join(packages)}")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install'] + packages,
            capture_output=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error installing packages: {e}")
        return False


def print_quick_start():
    """Print quick start instructions"""
    print("\n" + "="*70)
    print("QUICK START")
    print("="*70)
    
    print("""
To run the complete verification workflow:

  $ cd /users/epnyrk/Project/design/work/ProjectA/scripts
  $ python3 run_verification.py

This will:
  1. Generate golden model reference outputs
  2. Compare RTL simulation to reference (if available)
  3. Create visualization plots
  4. Generate detailed analysis reports

Prerequisites:
  ✓ This Python interpreter
  ✓ Required packages (checked above)
  ✓ Input image (checked above)
  ✓ RTL simulation outputs (for comparison only)

For more information:
  $ python3 run_verification.py --help
  $ cat README.md
    """)


def run_comprehensive_check():
    """Run all diagnostic checks"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " ROAD SIGN DETECTOR - SETUP & DIAGNOSTIC TOOL ".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    results = {}
    
    # Run all checks
    results['python_version'] = check_python_version()
    results['dependencies'] = check_dependencies()
    results['directories'] = check_directory_structure()
    results['scripts'] = check_script_files()
    results['input_image'] = check_input_image()
    results['rtl_outputs'] = check_rtl_outputs()
    results['disk_space'] = check_disk_space()
    
    # Print summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    critical = ['python_version', 'dependencies', 'directories', 'scripts', 'input_image']
    non_critical = ['rtl_outputs']
    
    critical_ok = all(results[k] for k in critical if k in results)
    rtl_ok = results.get('rtl_outputs', False)
    
    print("\nCritical Checks:")
    for check_name in critical:
        status = "✓ PASS" if results.get(check_name, False) else "✗ FAIL"
        print(f"  [{status}] {check_name.replace('_', ' ').title()}")
    
    print("\nNon-Critical Checks:")
    for check_name in non_critical:
        status = "✓ PASS" if results.get(check_name, False) else "✗ SKIP"
        print(f"  [{status}] {check_name.replace('_', ' ').title()}")
    
    print("\nDisk & System:")
    status = "✓ PASS" if results.get('disk_space', False) else "? WARN"
    print(f"  [{status}] Disk Space")
    
    # Overall status
    print("\n" + "="*70)
    
    if critical_ok:
        print("✓ ALL CRITICAL CHECKS PASSED")
        print("\nYou can now run:")
        print("  $ python3 golden_model_complete.py")
        
        if rtl_ok:
            print("  $ python3 comparison_and_analysis.py")
            print("  $ python3 detailed_report_generator.py")
        else:
            print("\nNote: Run RTL simulation first for comparison")
            print("      $ cd <rtl_directory> && make sim")
        
        print("\nOr run everything at once:")
        print("  $ python3 run_verification.py")
    else:
        print("✗ CRITICAL CHECKS FAILED")
        print("\nFix the issues above before proceeding")
        
        # Offer to install
        if not results.get('dependencies', False):
            response = input("\nWould you like to install missing packages? (y/n): ")
            if response.lower() == 'y':
                if install_dependencies():
                    print("✓ Packages installed successfully")
                    print("  Please run this script again to verify")
                else:
                    print("✗ Package installation failed")
    
    print("\n" + "="*70)
    print_quick_start()
    
    return critical_ok


def main():
    """Main entry point"""
    try:
        success = run_comprehensive_check()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
