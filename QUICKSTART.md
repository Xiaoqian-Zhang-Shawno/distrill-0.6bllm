# 快速开始指南

## 第一步：创建 Conda 环境

### Windows 用户

**方法 1: 使用脚本（推荐）**
```cmd
create_env.bat
```

**方法 2: 使用 Anaconda Prompt**
1. 打开 Anaconda Prompt
2. 导航到项目目录
3. 运行：
```cmd
conda create -n llmNano python=3.10 -y
conda activate llmNano
pip install -r requirements.txt
```

### Linux/Mac 用户

```bash
chmod +x setup_env.sh
./setup_env.sh
```

## 第二步：激活环境

**Windows:**
```cmd
conda activate llmNano
```

**Linux/Mac:**
```bash
conda activate llmNano
```

## 第三步：配置 API Key

API Key 已在 `config.py` 中配置，如需修改：

**Windows:**
```cmd
set DEEPSEEK_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

## 第四步：准备数据

创建训练数据文件 `./data/raw_data.jsonl`，格式如下：

```json
{"instruction": "请解释什么是机器学习？", "input": ""}
{"instruction": "翻译以下英文", "input": "Hello, how are you?"}
```

## 第五步：开始训练

**一键运行（推荐）:**
```bash
python run.py
```

**或分步执行:**
```bash
# 1. 生成教师模型回答
python prepare_data.py

# 2. 训练学生模型
python train.py
```

## 第六步：使用训练好的模型

```bash
python example_usage.py
```

## 常见问题

### Q: OpenMP 库冲突错误？

A: 如果遇到 `OMP: Error #15` 错误：
- **已自动修复**：代码已自动设置环境变量，直接运行即可
- 如果仍有问题，手动设置：`set KMP_DUPLICATE_LIB_OK=TRUE`
- 或使用 `fix_openmp.bat` 脚本

### Q: conda 命令未找到？

A: 
- Windows: 使用 Anaconda Prompt 而不是普通命令行
- 确保 Anaconda/Miniconda 已正确安装
- 检查是否添加到系统 PATH

### Q: 如何检查环境是否激活？

A: 运行 `python --version` 和 `conda info`，应该显示当前环境为 llmNano

### Q: 如何退出环境？

A: 运行 `conda deactivate`

### Q: 如何删除环境重新开始？

A: 
```bash
conda deactivate
conda env remove -n llmNano
```
然后重新运行 `create_env.bat` 或 `setup_env.sh`

## 下一步

- 查看 [README.md](README.md) 了解详细配置
- 查看 [ENV_SETUP.md](ENV_SETUP.md) 了解环境设置详情
