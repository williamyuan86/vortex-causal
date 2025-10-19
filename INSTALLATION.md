# Installation Guide | 安装指南

## Table of Contents | 目录

- [System Requirements | 系统要求](#system-requirements-系统要求)
- [Installation Methods | 安装方法](#installation-methods-安装方法)
- [Environment Setup | 环境配置](#environment-setup-环境配置)
- [Verification | 验证安装](#verification-验证安装)
- [Troubleshooting | 故障排除](#troubleshooting-故障排除)
- [Advanced Configuration | 高级配置](#advanced-configuration-高级配置)

## System Requirements | 系统要求

### Minimum Requirements | 最低要求

- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 4GB minimum (8GB+ recommended)
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Recommended Requirements | 推荐配置

- **Python**: 3.9 or 3.10
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 5GB free space
- **CPU**: Multi-core processor (4+ cores)

## Installation Methods | 安装方法

### Method 1: Standard Installation (Recommended) | 标准安装（推荐）

#### Step 1: Clone Repository | 克隆仓库

```bash
git clone https://github.com/your-org/vortex-causal.git
cd vortex-causal
```

#### Step 2: Create Virtual Environment | 创建虚拟环境

**Windows | Windows系统:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux | macOS/Linux系统:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies | 安装依赖

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Method 2: Docker Installation | Docker安装

#### Using Pre-built Image | 使用预构建镜像

```bash
# Pull the image
docker pull vortex-causal:latest

# Run the container
docker run -it --rm vortex-causal python main.py
```

#### Building from Source | 从源码构建

```bash
# Build the image
docker build -t vortex-causal .

# Run with volume mount for data
docker run -it --rm -v $(pwd)/data:/app/data vortex-causal
```

### Method 3: Conda Installation | Conda安装

```bash
# Create conda environment
conda create -n vortex-causal python=3.9
conda activate vortex-causal

# Install dependencies
conda install numpy pandas scikit-learn pytorch torchvision -c pytorch
pip install -r requirements.txt

# Install package
pip install -e .
```

## Environment Setup | 环境配置

### GPU Support | GPU支持

If you have an NVIDIA GPU, install CUDA-enabled PyTorch:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA (replace cu118 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### LLM API Configuration | LLM API配置

For full LLM integration, configure API keys:

```bash
# OpenAI API
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic Claude API
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Or create .env file
echo "OPENAI_API_KEY=your-key" > .env
echo "ANTHROPIC_API_KEY=your-key" >> .env
```

### Environment Variables | 环境变量

Create a `.env` file in the project root:

```bash
# LLM API Keys
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Optional: Model preferences
DEFAULT_CAUSAL_MODEL=gpt-4
DEFAULT_ENSEMBLE_MODEL=claude-3-sonnet

# Optional: Hardware settings
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
```

### Development Environment | 开发环境

For developers, install additional tools:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Configure IDE (optional)
# VS Code: Install Python extension and Jupyter extension
```

## Verification | 验证安装

### Quick Test | 快速测试

```bash
# Run basic functionality test
python -c "from src.utils.data_loader import load_sachs; print('✅ Basic import works')"

# Run test suite
python -m pytest src/tests/ -v

# Run example
python examples/basic_discovery.py
```

### Expected Output | 预期输出

You should see:

```
✅ Basic import works
============================= test session starts ==============================
collected 64 items

src/tests/test_algorithm_selector.py .......                          [ 10%]
...
============================== 64 passed in 15.40s ==========================
✅ All tests passed!
```

### Performance Benchmark | 性能基准

```bash
# Run quick benchmark
python scripts/run_benchmarks.py --n-datasets 2

# Expected: Should complete without errors and generate benchmark report
```

## Troubleshooting | 故障排除

### Common Issues | 常见问题

#### Issue 1: PyTorch CUDA Error | PyTorch CUDA错误

**Problem | 问题:**
```
RuntimeError: CUDA out of memory
```

**Solution | 解决方案:**
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""

# Or reduce batch size in configuration
```

#### Issue 2: Import Errors | 导入错误

**Problem | 问题:**
```
ModuleNotFoundError: No module named 'src.utils.data_loader'
```

**Solution | 解决方案:**
```bash
# Ensure you're in the project directory
cd /path/to/vortex-causal

# Install in development mode
pip install -e .

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Issue 3: LLM API Errors | LLM API错误

**Problem | 问题:**
```
OpenAI API key not found
```

**Solution | 解决方案:**
```bash
# Set environment variable
export OPENAI_API_KEY="your-key-here"

# Or create .env file
echo "OPENAI_API_KEY=your-key" > .env
```

#### Issue 4: Memory Issues | 内存问题

**Problem | 问题:**
```
MemoryError or slow performance
```

**Solution | 解决方案:**
```bash
# Reduce dataset size in config
# Use smaller models
# Enable memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Platform-Specific Issues | 平台特定问题

#### Windows | Windows系统

```bash
# If you get DLL errors, install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# For long path issues, enable long paths in Windows
# or move project to shorter path like C:\vc\
```

#### macOS | macOS系统

```bash
# Install Xcode command line tools
xcode-select --install

# If you have M1/M2 Mac, ensure PyTorch supports MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

#### Linux | Linux系统

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev build-essential

# For CUDA, install NVIDIA drivers
sudo apt-get install nvidia-driver-470
```

## Advanced Configuration | 高级配置

### Custom Configuration | 自定义配置

Create `config/custom.yaml`:

```yaml
# Model settings
models:
  default_causal_model: "gpt-4"
  ensemble_models: ["gpt-4", "claude-3-sonnet", "qwen-max"]

# Algorithm settings
algorithm_selector:
  thresh_hard: 0.8
  thresh_ambiguous: 0.3

# Graph generation
graph_generation:
  lambda1: 0.1
  max_iter: 100
  lr: 0.01

# Hardware settings
hardware:
  use_gpu: true
  batch_size: 32
  num_workers: 4
```

### Distributed Training | 分布式训练

For multi-GPU training:

```bash
# Set up multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export RANK=0

# Run distributed training
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  main.py --distributed
```

### Production Deployment | 生产部署

For production use:

```bash
# Create production environment
conda create -n vortex-prod python=3.9
conda activate vortex-prod

# Install only production dependencies
pip install numpy pandas scikit-learn torch networkx optuna

# Set up logging
export LOG_LEVEL=INFO
export LOG_FILE=/var/log/vortex-causal.log

# Run with process manager
pm2 start ecosystem.config.js
```

---

## 中文版本 (Chinese Version)

### 系统要求

- **Python**: 3.8 或更高版本（推荐 3.9+）
- **内存**: 最低 4GB（推荐 8GB+）
- **存储**: 2GB 可用空间
- **操作系统**: Windows 10+、macOS 10.14+ 或 Linux (Ubuntu 18.04+)

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-org/vortex-causal.git
cd vortex-causal
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
pip install -e .
```

### 验证安装

```bash
python -m pytest src/tests/ -v
python examples/basic_discovery.py
```

### 常见问题解决

**导入错误**: 确保在项目根目录并运行 `pip install -e .`

**CUDA错误**: 检查GPU驱动和PyTorch CUDA版本兼容性

**API密钥错误**: 设置环境变量或创建 `.env` 文件

详细帮助请查看 [故障排除指南](docs/troubleshooting.md)。