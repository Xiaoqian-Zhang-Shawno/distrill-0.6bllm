"""
配置文件
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TeacherConfig:
    """教师模型配置（DeepSeek API）"""
    api_key: str = os.getenv(
        "DEEPSEEK_API_KEY", "sk-eceb580e89904659922c04f13d1fa9b7")
    base_url: str = "https://api.deepseek.com/v1"
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class StudentConfig:
    """学生模型配置（Qwen2.5-0.5B）"""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    cache_dir: Optional[str] = None


@dataclass
class LoRAConfig:
    """QLoRA 配置"""
    r: int = 64  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha
    target_modules: list = None  # 将在 train.py 中设置
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            # Qwen2.5 的默认目标模块
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]


@dataclass
class QuantizationConfig:
    """推理/导出量化配置"""
    mode: str = "fp16"  # 可选：int4 / int8 / fp16
    use_double_quant: bool = True
    quant_type: str = "nf4"  # 仅 int4 时使用：nf4 或 fp4
    compute_dtype: str = "float16"  # 计算精度


@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    fp16: bool = True
    bf16: bool = False
    optim: str = "adamw_torch"  # 使用标准 adamw，避免 bitsandbytes 优化器问题
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    dataset_path: str = "./data/train_dataset.jsonl"
    eval_dataset_path: Optional[str] = None
    # 验证集配置
    eval_split_ratio: float = 0.1  # 验证集比例（10%）
    eval_split_seed: int = 42  # 随机种子，确保可复现
    # 导出配置
    quant_mode: str = "int4"  # int4 / int8 / fp16
    onnx_export_path: str = "./export/model.onnx"
    mnn_export_path: str = "./export/model.mnn"