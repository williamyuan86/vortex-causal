# Vortex-Causal Test Suite

This directory contains the complete test suite for the Vortex-Causal causal discovery system.

## Test Files

### Core Tests

#### `test_autoeval_controller.py`
- **Purpose**: Comprehensive unit tests for the AutoEval controller class
- **Coverage**:
  - Controller initialization
  - Configuration evaluation
  - Adaptive refinement logic
  - Iteration control
  - Stopping conditions
  - Error handling
  - Full optimization loops
- **Usage**: `python test_autoeval_controller.py`

### Performance Tests

#### `test_performance_sachs.py`
- **Purpose**: Performance evaluation on the Sachs protein signaling dataset
- **Coverage**:
  - Algorithm selection
  - Baseline vs constrained NOTEARS comparison
  - Graph metrics evaluation (SHD, F1, AUC-ROC, AUC-PR)
  - Causal effect estimation
- **Usage**: `python test_performance_sachs.py`

#### `test_comprehensive_performance.py`
- **Purpose**: Multi-configuration performance testing
- **Coverage**:
  - Multiple parameter configurations
  - Performance comparison across settings
  - Detailed benchmark reporting
  - Statistical analysis
- **Usage**: `python test_comprehensive_performance.py`

### Test Utilities

#### `run_all_tests.py`
- **Purpose**: Unified test runner for all test files
- **Features**:
  - Runs all tests with timeout protection
  - Generates summary reports
  - Captures error details
  - Saves results to file
- **Usage**: `python run_all_tests.py`

## Test Categories

### 1. Unit Tests
- Individual component testing
- Mocking and isolation
- Edge case validation
- Error condition testing

### 2. Integration Tests
- End-to-end pipeline testing
- Component interaction validation
- Real dataset testing

### 3. Performance Tests
- Algorithm benchmarking
- Configuration optimization
- Scalability testing
- Statistical validation

## Running Tests

### Run All Tests
```bash
cd src/tests
python run_all_tests.py
```

### Run Individual Tests
```bash
# AutoEval controller tests
python test_autoeval_controller.py

# Sachs dataset performance
python test_performance_sachs.py

# Comprehensive performance analysis
python test_comprehensive_performance.py
```

### Run with Specific Environment
```bash
# Using project virtual environment
vortex_env/Scripts/python.exe src/tests/run_all_tests.py
```

## Test Results

Test results are saved to:
- `test_results_report.txt` - Summary report (project root)
- `output/` - AutoEval controller results
- `results/` - Performance test results

## Test Coverage

The test suite covers:

- ✅ **AutoEval Controller**: 100% core functionality
- ✅ **Algorithm Selection**: Integration testing
- ✅ **Graph Learning**: Performance validation
- ✅ **Constraint Fusion**: End-to-end testing
- ✅ **Metrics Calculation**: Accuracy verification
- ✅ **Error Handling**: Edge case coverage
- ✅ **Performance**: Real-world dataset validation

## Contributing

When adding new tests:

1. Follow the naming convention: `test_<module>_<functionality>.py`
2. Include comprehensive error handling
3. Add documentation for test purpose and coverage
4. Update this README file
5. Ensure tests run independently

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in the Python path
2. **Timeout**: Increase timeout in `run_all_tests.py` for long-running tests
3. **Data Loading**: Verify data files exist in `data/` directory
4. **Environment**: Use the project virtual environment

### Debug Mode
Run tests with verbose logging:
```bash
python -u test_autoeval_controller.py
```

---

*Last updated: 2025-10-19*