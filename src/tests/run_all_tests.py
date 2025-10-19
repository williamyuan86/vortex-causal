#!/usr/bin/env python3
"""
Comprehensive test runner for Vortex-Causal test suite.

This script runs all available tests and provides a summary report.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test_file(test_file: Path) -> Tuple[bool, str, float]:
    """
    Run a single test file and capture results.

    Args:
        test_file: Path to the test file

    Returns:
        Tuple of (success, output, runtime)
    """
    start_time = time.time()

    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        runtime = time.time() - start_time
        success = result.returncode == 0
        output = result.stdout + result.stderr

        return success, output, runtime

    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        return False, "Test timed out after 5 minutes", runtime
    except Exception as e:
        runtime = time.time() - start_time
        return False, f"Error running test: {str(e)}", runtime

def main():
    """
    Main test runner function.
    """
    logger.info("=== Vortex-Causal Test Suite Runner ===")

    # Find all test files
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / "test_autoeval_controller.py",
        test_dir / "test_performance_sachs.py",
        test_dir / "test_comprehensive_performance.py",
    ]

    # Add root level test file if it exists
    root_test = project_root / "test_autoeval.py"
    if root_test.exists():
        test_files.append(root_test)

    logger.info(f"Found {len(test_files)} test files")

    # Run tests
    results = []
    total_start_time = time.time()

    for test_file in test_files:
        logger.info(f"\n--- Running {test_file.name} ---")

        success, output, runtime = run_test_file(test_file)

        results.append({
            'file': test_file.name,
            'success': success,
            'runtime': runtime,
            'output': output
        })

        status = "PASS" if success else "FAIL"
        logger.info(f"{test_file.name}: {status} ({runtime:.2f}s)")

        # Show first few lines of output for debugging
        if not success:
            output_lines = output.split('\n')[:10]
            logger.error("Error output:")
            for line in output_lines:
                if line.strip():
                    logger.error(f"  {line}")

    total_runtime = time.time() - total_start_time

    # Generate summary report
    logger.info("\n" + "="*60)
    logger.info("TEST SUITE SUMMARY")
    logger.info("="*60)

    passed = sum(1 for r in results if r['success'])
    total = len(results)

    logger.info(f"Total tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Total runtime: {total_runtime:.2f}s")

    # Detailed results
    logger.info("\nDetailed Results:")
    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        logger.info(f"  {result['file']:<30} {status:<6} ({result['runtime']:.2f}s)")

    # Failed tests details
    failed_tests = [r for r in results if not r['success']]
    if failed_tests:
        logger.info("\nFailed Tests Details:")
        for test in failed_tests:
            logger.info(f"\n{test['file']}:")
            # Show relevant error lines
            error_lines = [line for line in test['output'].split('\n')
                          if any(keyword in line.lower()
                                for keyword in ['error', 'failed', 'exception', 'traceback'])]
            for line in error_lines[:5]:  # First 5 error lines
                if line.strip():
                    logger.info(f"  {line}")

    # Save report to file
    report_file = project_root / "test_results_report.txt"
    with open(report_file, 'w') as f:
        f.write("Vortex-Causal Test Suite Report\n")
        f.write("="*40 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total tests: {total}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {total - passed}\n")
        f.write(f"Total runtime: {total_runtime:.2f}s\n\n")

        for result in results:
            status = "PASS" if result['success'] else "FAIL"
            f.write(f"{result['file']:<30} {status:<6} ({result['runtime']:.2f}s)\n")

    logger.info(f"\nDetailed report saved to: {report_file}")

    # Return overall success
    return all(r['success'] for r in results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)