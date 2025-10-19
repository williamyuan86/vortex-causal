#!/usr/bin/env python3
"""
Environment Verification Script
环境验证脚本
"""

import sys
import traceback

def test_imports():
    """Test all necessary imports"""
    print("=" * 60)
    print("Testing Library Imports | 测试库导入")
    print("=" * 60)

    # Core scientific libraries
    try:
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        print("[OK] Core scientific libraries imported")
        # Make numpy available globally for this script
        globals()['np'] = np
    except Exception as e:
        print(f"[ERROR] Core scientific libraries: {e}")
        return False

    # Deep learning
    try:
        import torch
        import torchvision
        print(f"[OK] PyTorch {torch.__version__} imported")
    except Exception as e:
        print(f"[ERROR] PyTorch: {e}")
        return False

    # Graph and optimization
    try:
        import networkx as nx
        import optuna
        print("[OK] NetworkX and Optuna imported")
    except Exception as e:
        print(f"[ERROR] NetworkX/Optuna: {e}")
        return False

    # Visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("[OK] Matplotlib and Seaborn imported")
    except Exception as e:
        print(f"[ERROR] Visualization libraries: {e}")
        return False

    # Vortex-Causal modules
    try:
        from src.utils.data_loader import create_synthetic_dataset, load_sachs
        from src.selector.algorithm_selector import run_selector
        from src.constraint_fusion.fusion import apply_hard_constraints
        from src.graph_generation.notears_torch import train_notears_torch
        from src.effect_estimation.estimator import ols_ate, ipw_ate
        print("[OK] All Vortex-Causal modules imported")
    except Exception as e:
        print(f"[ERROR] Vortex-Causal modules: {e}")
        traceback.print_exc()
        return False

    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\n" + "=" * 60)
    print("Testing Basic Functionality | 测试基本功能")
    print("=" * 60)

    try:
        # Test synthetic data generation
        from src.utils.data_loader import create_synthetic_dataset
        df, adj = create_synthetic_dataset(n_samples=100, n_vars=5, random_state=42)
        print(f"[OK] Synthetic dataset generated: {df.shape}")

        # Test algorithm selector
        from src.selector.algorithm_selector import run_selector
        selector_result = run_selector(df, thresh_hard=0.8, thresh_ambiguous=0.3)
        print(f"[OK] Algorithm selector completed: {len(selector_result['hard'])} hard edges")

        # Test constraint fusion
        from src.constraint_fusion.fusion import apply_hard_constraints
        mask_fixed, W_fixed = apply_hard_constraints(np.zeros((5, 5)), selector_result['hard'])
        print(f"[OK] Constraint fusion completed")

        # Test effect estimation
        from src.effect_estimation.estimator import ols_ate, ipw_ate
        # Create simple treatment/outcome data
        df_test = df.copy()
        df_test['treatment'] = (df_test.iloc[:, -1] > df_test.iloc[:, -1].median()).astype(int)
        df_test['outcome'] = df_test.iloc[:, 0]

        ate_ols = ols_ate(df_test, 'treatment', 'outcome', [df_test.columns[1]])
        print(f"[OK] Effect estimation completed: OLS ATE = {ate_ols:.4f}")

        return True

    except Exception as e:
        print(f"[ERROR] Basic functionality: {e}")
        traceback.print_exc()
        return False

def test_gpu_availability():
    """Test GPU availability (optional)"""
    print("\n" + "=" * 60)
    print("Testing GPU Availability | 测试GPU可用性")
    print("=" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"[OK] CUDA version: {torch.version.cuda}")
        else:
            print("[INFO] CUDA not available - using CPU")

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("[OK] MPS available (Apple Silicon)")

        return True

    except Exception as e:
        print(f"[ERROR] GPU check: {e}")
        return False

def main():
    """Main verification function"""
    print("VORTEX-CAUSAL ENVIRONMENT VERIFICATION")
    print("VORTEX-CAUSAL 环境验证")
    print("Python version:", sys.version)
    print()

    all_passed = True

    # Run all tests
    all_passed &= test_imports()
    all_passed &= test_basic_functionality()
    all_passed &= test_gpu_availability()

    # Final result
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT | 验证结果")
    print("=" * 60)

    if all_passed:
        print("[SUCCESS] All tests passed! Environment is ready.")
        print("[成功] 所有测试通过！环境已准备就绪。")
        print("\nNext steps | 下一步:")
        print("1. Run the full test suite: python -m pytest src/tests/ -v")
        print("2. Try the main program: python main.py")
        print("3. Run benchmarks: python scripts/run_benchmarks.py --n-datasets 2")
        return 0
    else:
        print("[FAILURE] Some tests failed. Please check the errors above.")
        print("[失败] 部分测试失败。请检查上述错误。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)