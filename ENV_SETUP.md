# Conda 环境设置指南

本项目使用 conda 虚拟环境 `llmNano` 来管理依赖。

## 快速开始

### Windows 用户

1. **自动设置（推荐）**
   ```cmd
   setup_env.bat
   ```
   这会自动创建环境并安装所有依赖。

2. **手动设置**
   ```cmd
   REM 创建环境
   conda create -n llmNano python=3.10 -y
   
   REM 激活环境
   conda activate llmNano
   
   REM 安装依赖
   pip install -r requirements.txt
   ```

3. **激活环境**
   - 方法 1: 运行 `activate_env.bat`
   - 方法 2: 在命令行中运行 `conda activate llmNano`

### Linux/Mac 用户

1. **自动设置（推荐）**
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

2. **手动设置**
   ```bash
   # 创建环境
   conda create -n llmNano python=3.10 -y
   
   # 激活环境
   conda activate llmNano
   
   # 安装依赖
   pip install -r requirements.txt
   ```

## 使用环境

### 激活环境

**Windows:**
```cmd
conda activate llmNano
```

**Linux/Mac:**
```bash
conda activate llmNano
```

### 运行项目

激活环境后，可以运行项目脚本：

```bash
# 准备数据并训练
python run.py

# 或分步执行
python prepare_data.py
python train.py

# 使用训练好的模型
python example_usage.py
```

### 退出环境

```bash
conda deactivate
```

## 环境信息

- **环境名称**: llmNano
- **Python 版本**: 3.10
- **主要依赖**:
  - torch >= 2.0.0
  - transformers >= 4.40.0
  - peft >= 0.8.0
  - datasets >= 2.14.0
  - accelerate >= 0.27.0
  - bitsandbytes >= 0.41.0
  - openai >= 1.12.0

## 常见问题

### 1. conda 命令未找到

**解决方案:**
- 确保已安装 Anaconda 或 Miniconda
- 将 conda 添加到系统 PATH
- 或使用 Anaconda Prompt (Windows)

### 2. 环境已存在

如果环境已存在，可以：
- 删除旧环境: `conda env remove -n llmNano`
- 或直接激活: `conda activate llmNano`

### 3. 依赖安装失败

**解决方案:**
- 检查网络连接
- 升级 pip: `python -m pip install --upgrade pip`
- 使用国内镜像（如需要）:
  ```bash
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

### 4. CUDA 相关问题

如果遇到 CUDA 相关错误：
- 确保已安装对应版本的 CUDA
- 检查 PyTorch 版本是否与 CUDA 版本匹配
- 参考 [PyTorch 官网](https://pytorch.org/) 安装正确的版本

## 更新依赖

如果需要更新依赖：

```bash
# 激活环境
conda activate llmNano

# 更新 requirements.txt 中的版本号后
pip install -r requirements.txt --upgrade
```

## 导出环境

导出环境配置（可选）：

```bash
conda env export > environment.yml
```

恢复环境：

```bash
conda env create -f environment.yml
```
