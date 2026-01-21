#!/bin/bash
# Conda 环境设置脚本 (Linux/Mac)

echo "========================================"
echo "设置 llmNano Conda 环境"
echo "========================================"

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda 命令"
    echo "请确保已安装 Anaconda 或 Miniconda，并将其添加到 PATH"
    exit 1
fi

echo ""
echo "步骤 1: 创建 conda 环境 llmNano (Python 3.10)"
conda create -n llmNano python=3.10 -y

if [ $? -ne 0 ]; then
    echo "错误: 创建环境失败"
    exit 1
fi

echo ""
echo "步骤 2: 激活环境并安装依赖"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmNano

if [ $? -ne 0 ]; then
    echo "错误: 激活环境失败"
    exit 1
fi

echo ""
echo "步骤 3: 升级 pip"
python -m pip install --upgrade pip

echo ""
echo "步骤 4: 安装项目依赖"
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "错误: 安装依赖失败"
    exit 1
fi

echo ""
echo "========================================"
echo "环境设置完成！"
echo "========================================"
echo ""
echo "使用以下命令激活环境:"
echo "  conda activate llmNano"
echo ""
