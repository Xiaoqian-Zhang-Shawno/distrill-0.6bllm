"""
数据准备脚本
用于准备训练数据
"""
import json
import os
import random
from typing import List, Dict, Tuple
from teacher_model import TeacherModel
from config import TeacherConfig, TrainingConfig


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """保存为 JSONL 文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_dataset(data: List[Dict], eval_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    分割数据集为训练集和验证集
    
    Args:
        data: 原始数据列表
        eval_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        (train_data, eval_data): 训练集和验证集
    """
    random.seed(seed)
    data_shuffled = data.copy()
    random.shuffle(data_shuffled)
    
    total = len(data_shuffled)
    eval_size = int(total * eval_ratio)
    
    eval_data = data_shuffled[:eval_size]
    train_data = data_shuffled[eval_size:]
    
    return train_data, eval_data


def prepare_training_data(input_file: str, output_file: str, 
                          teacher_config: TeacherConfig = None,
                          eval_output_file: str = None,
                          eval_ratio: float = 0.1,
                          eval_seed: int = 42):
    """
    准备训练数据（包括训练集和验证集）
    
    Args:
        input_file: 输入数据文件路径（JSONL 格式）
        output_file: 训练集输出文件路径（JSONL 格式）
        teacher_config: 教师模型配置
        eval_output_file: 验证集输出文件路径（可选）
        eval_ratio: 验证集比例（默认 0.1，即 10%）
        eval_seed: 随机种子
    """
    print(f"加载输入数据: {input_file}")
    input_data = load_jsonl(input_file)
    print(f"共加载 {len(input_data)} 条数据")
    
    # 检查是否已生成数据
    train_exists = os.path.exists(output_file)
    eval_exists = eval_output_file and os.path.exists(eval_output_file)
    
    if train_exists:
        print(f"检查已存在的训练数据文件: {output_file}")
        existing_train = load_jsonl(output_file)
        if existing_train and "teacher_output" in existing_train[0]:
            print(f"发现已生成的训练数据，共 {len(existing_train)} 条")
            train_data = existing_train
        else:
            train_data = None
    else:
        train_data = None
    
    if eval_output_file and eval_exists:
        print(f"检查已存在的验证数据文件: {eval_output_file}")
        existing_eval = load_jsonl(eval_output_file)
        if existing_eval and "teacher_output" in existing_eval[0]:
            print(f"发现已生成的验证数据，共 {len(existing_eval)} 条")
            eval_data = existing_eval
        else:
            eval_data = None
    else:
        eval_data = None
    
    # 如果数据都已存在，直接返回
    if train_data is not None and (not eval_output_file or eval_data is not None):
        return train_data, eval_data
    
    # 使用教师模型生成回答
    if teacher_config is None:
        teacher_config = TeacherConfig()
    
    teacher = TeacherModel(teacher_config)
    
    # 如果训练集不存在，需要生成
    if train_data is None:
        # 分割数据集
        if eval_output_file and eval_data is None:
            print(f"\n分割数据集: 训练集 {1-eval_ratio:.1%}, 验证集 {eval_ratio:.1%}")
            train_raw, eval_raw = split_dataset(input_data, eval_ratio, eval_seed)
            print(f"训练集: {len(train_raw)} 条, 验证集: {len(eval_raw)} 条")
        else:
            train_raw = input_data
            eval_raw = []
        
        # 生成训练集
        print(f"\n生成训练集回答...")
        teacher.generate_dataset(train_raw, output_file)
        train_data = load_jsonl(output_file)
    
    # 如果验证集不存在且需要生成
    if eval_output_file and eval_data is None:
        if not train_exists:  # 如果训练集刚生成，使用之前分割的验证集
            print(f"\n生成验证集回答...")
            teacher.generate_dataset(eval_raw, eval_output_file)
        else:  # 如果训练集已存在，需要重新分割
            print(f"\n重新分割数据集生成验证集...")
            _, eval_raw = split_dataset(input_data, eval_ratio, eval_seed)
            print(f"验证集: {len(eval_raw)} 条")
            teacher.generate_dataset(eval_raw, eval_output_file)
        eval_data = load_jsonl(eval_output_file)
    
    return train_data, eval_data


if __name__ == "__main__":
    # 示例：准备数据
    from config import TrainingConfig
    
    training_config = TrainingConfig()
    input_file = "./data/raw_data.jsonl"
    output_file = training_config.dataset_path
    eval_output_file = training_config.eval_dataset_path or "./data/eval_dataset.jsonl"
    
    # 如果输入文件不存在，创建一个示例文件
    if not os.path.exists(input_file):
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        example_data = [
            {
                "instruction": "请解释什么是机器学习？",
                "input": ""
            },
            {
                "instruction": "写一首关于春天的诗",
                "input": ""
            },
            {
                "instruction": "翻译以下英文",
                "input": "Hello, how are you?"
            }
        ]
        save_jsonl(example_data, input_file)
        print(f"已创建示例数据文件: {input_file}")
    
    # 准备训练数据（包括训练集和验证集）
    train_data, eval_data = prepare_training_data(
        input_file, 
        output_file,
        eval_output_file=eval_output_file,
        eval_ratio=training_config.eval_split_ratio,
        eval_seed=training_config.eval_split_seed
    )
    
    print(f"\n✓ 训练集: {len(train_data)} 条")
    if eval_data:
        print(f"✓ 验证集: {len(eval_data)} 条")

