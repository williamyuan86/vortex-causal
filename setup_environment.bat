@echo off
echo ========================================
echo Vortex-Causal Environment Setup Script
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo Python found successfully!
echo.

echo [2/4] Creating virtual environment...
if exist vortex_env (
    echo Removing existing virtual environment...
    rmdir /s /q vortex_env
)
python -m venv vortex_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)
echo Virtual environment created successfully!
echo.

echo [3/4] Activating virtual environment and installing dependencies...
call vortex_env\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing core dependencies...
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scipy>=1.7.0
pip install scikit-learn>=1.0.0

echo Installing PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo Installing additional dependencies...
pip install networkx>=2.6.0
pip install optuna>=3.0.0
pip install tqdm>=4.62.0
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0
pip install statsmodels>=0.12.0

echo Installing testing and development tools...
pip install pytest>=6.2.0
pip install pytest-cov>=2.12.0

echo [4/4] Verifying installation...
echo.
echo Testing basic imports...
python -c "import numpy, pandas, torch, sklearn; print('✅ Core libraries imported successfully')"

echo Testing Vortex-Causal imports...
python -c "from src.utils.data_loader import create_synthetic_dataset; print('✅ Vortex-Causal imports successful')"

echo Running basic test...
python -m pytest src/tests/test_algorithm_selector.py::TestAlgorithmSelector::test_run_selector_output_structure -v

echo.
echo ========================================
echo SETUP COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo vortex_env\Scripts\activate
echo.
echo To run the main program:
echo python main.py
echo.
echo To run tests:
echo python -m pytest src/tests/ -v
echo.

pause