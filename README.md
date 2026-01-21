# 知识蒸馏项目：DeepSeek32B → Qwen2.5-0.5B-Instruct

使用 DeepSeek32B (API) 作为教师模型，通过知识蒸馏训练 Qwen2.5-0.5B-Instruct 学生模型，使用 QLoRA 进行高效微调。

## 项目结构

```
.
├── train.py              # 主训练脚本
├── teacher_model.py      # 教师模型 API 调用模块
├── prepare_data.py       # 数据准备脚本
├── config.py             # 配置文件
├── requirements.txt      # 依赖包
├── run.py                # 快速启动脚本
├── example_usage.py      # 使用示例
├── setup_env.bat         # Windows 环境设置脚本
├── setup_env.sh          # Linux/Mac 环境设置脚本
├── activate_env.bat       # Windows 环境激活脚本
├── ENV_SETUP.md          # 环境设置详细说明
└── README.md            # 本文件
```

## 环境设置

### 1. 创建 Conda 虚拟环境（推荐）

本项目使用 conda 虚拟环境 `llmNano` 来管理依赖。

**Windows 用户:**
```cmd
REM 自动设置（推荐）
setup_env.bat

REM 或手动设置
conda create -n llmNano python=3.10 -y
conda activate llmNano
pip install -r requirements.txt
```

**Linux/Mac 用户:**
```bash
# 自动设置（推荐）
chmod +x setup_env.sh
./setup_env.sh

# 或手动设置
conda create -n llmNano python=3.10 -y
conda activate llmNano
pip install -r requirements.txt
```

详细说明请参考 [ENV_SETUP.md](ENV_SETUP.md)

### 2. 设置 DeepSeek API Key

设置环境变量：

```bash
# Windows
set DEEPSEEK_API_KEY=your_api_key_here

# Linux/Mac
export DEEPSEEK_API_KEY=your_api_key_here
```

或者在 `config.py` 中直接设置。

## 使用方法

### 步骤 1: 准备数据

准备你的训练数据，格式为 JSONL，每行一个 JSON 对象：

```json
{"instruction": "请解释什么是机器学习？", "input": ""}
{"instruction": "翻译以下英文", "input": "Hello, how are you?"}
```

将数据保存到 `./data/raw_data.jsonl`

### 步骤 2: 生成教师模型回答

**确保已激活 conda 环境:**
```bash
conda activate llmNano
```

运行数据准备脚本，使用 DeepSeek API 生成教师模型的回答（包括训练集和验证集）：

```bash
python prepare_data.py
```

或者手动调用：

```python
from prepare_data import prepare_training_data
from config import TeacherConfig, TrainingConfig

teacher_config = TeacherConfig()
training_config = TrainingConfig()

train_data, eval_data = prepare_training_data(
    input_file="./data/raw_data.jsonl",
    output_file="./data/train_dataset.jsonl",
    teacher_config=teacher_config,
    eval_output_file="./data/eval_dataset.jsonl",
    eval_ratio=0.1,  # 验证集比例 10%
    eval_seed=42
)
```

这将生成：
- 训练数据集：`./data/train_dataset.jsonl`（默认 90% 的数据）
- 验证数据集：`./data/eval_dataset.jsonl`（默认 10% 的数据）

验证集比例可在 `config.py` 中的 `TrainingConfig.eval_split_ratio` 配置（默认 0.1，即 10%）。

### 步骤 3: 开始训练

**确保已激活 conda 环境:**
```bash
conda activate llmNano
```

运行训练脚本：

```bash
python train.py
```

或者使用快速启动脚本（自动执行所有步骤）：

```bash
python run.py
```

训练过程会自动：
- 加载 Qwen2.5-0.5B-Instruct 模型
- 设置 QLoRA 配置
- 使用教师模型的回答作为监督信号进行训练
- 保存训练好的模型到 `./output` 目录

## 配置说明

### 教师模型配置 (TeacherConfig)

在 `config.py` 中修改：

- `api_key`: DeepSeek API Key
- `base_url`: API 基础 URL（默认：https://api.deepseek.com/v1）
- `model_name`: 模型名称（默认：deepseek-chat）
- `temperature`: 生成温度（默认：0.7）
- `max_tokens`: 最大生成 token 数（默认：2048）

### 学生模型配置 (StudentConfig)

- `model_name`: 学生模型名称（默认：Qwen/Qwen2.5-0.5B-Instruct）
- `cache_dir`: 模型缓存目录（可选）

### QLoRA 配置 (LoRAConfig)

- `r`: LoRA rank（默认：64）
- `lora_alpha`: LoRA alpha（默认：16）
- `target_modules`: 目标模块（默认：["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]）
- `lora_dropout`: Dropout 率（默认：0.1）

### 训练配置 (TrainingConfig)

- `output_dir`: 输出目录（默认：./output）
- `num_train_epochs`: 训练轮数（默认：3）
- `per_device_train_batch_size`: 每设备训练批次大小（默认：4）
- `learning_rate`: 学习率（默认：2e-4）
- `max_seq_length`: 最大序列长度（默认：2048）
- `dataset_path`: 训练数据集路径（默认：./data/train_dataset.jsonl）
- `eval_dataset_path`: 验证数据集路径（默认：None，会自动生成）
- `eval_split_ratio`: 验证集比例（默认：0.1，即 10%）
- `eval_split_seed`: 数据分割随机种子（默认：42，确保可复现）

## 输出

训练完成后，模型会保存到 `./output` 目录，包括：
- LoRA 适配器权重
- 分词器文件
- 训练日志

## 使用训练好的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, "./output")

# 使用模型
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

## 注意事项

1. **API 限流**: 生成教师模型回答时，代码会自动添加延迟以避免 API 限流。如果遇到限流问题，可以增加延迟时间。

2. **内存要求**: 使用 4-bit 量化后，Qwen2.5-0.5B 模型可以在较小的 GPU 上运行（如 8GB VRAM）。

3. **数据格式**: 确保输入数据包含 `instruction` 字段，可选 `input` 字段。

4. **训练时间**: 根据数据集大小和硬件配置，训练时间会有所不同。

## 故障排除

### 问题：OpenMP 库冲突（Windows）

如果遇到以下错误：
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**解决方案：**
- **方法 1（推荐）**：代码已自动处理，直接运行即可
- **方法 2**：手动设置环境变量
  ```cmd
  set KMP_DUPLICATE_LIB_OK=TRUE
  python run.py
  ```
- **方法 3**：使用 `fix_openmp.bat` 脚本
  ```cmd
  fix_openmp.bat python run.py
  ```

### 问题：API 调用失败
- 检查 API Key 是否正确设置
- 检查网络连接
- 确认 API 余额充足

### 问题：CUDA 内存不足
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 减小 `max_seq_length`

### 问题：模型加载失败
- 检查网络连接（需要下载模型）
- 确认 transformers 版本 >= 4.40.0

## 许可证

本项目遵循各模型和库的相应许可证。

