"""
知识蒸馏训练脚本
使用 DeepSeek32B (API) 作为教师模型，Qwen2.5-0.5B-Instruct 作为学生模型
使用 QLoRA 进行微调
"""
import os
import json

# 解决 Windows OpenMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from config import StudentConfig, LoRAConfig, TrainingConfig
from prepare_data import load_jsonl


def load_model_and_tokenizer(config: StudentConfig):
    """加载模型和分词器"""
    print(f"加载模型: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        cache_dir=config.cache_dir
    )

    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型（使用 4-bit 量化以节省内存）
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=config.cache_dir,
        quantization_config=quantization_config,
    )

    return model, tokenizer


def setup_lora(model, config: LoRAConfig):
    """设置 QLoRA"""
    print("设置 QLoRA...")

    # 准备模型用于 k-bit 训练
    model = prepare_model_for_kbit_training(model)

    # LoRA 配置
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type,
    )

    # 应用 LoRA
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    model.print_trainable_parameters()

    return model


def format_prompt(instruction: str, input_text: str = "", output: str = ""):
    """
    格式化提示（Qwen2.5-Instruct 格式）
    """
    if input_text:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    if output:
        prompt += output

    return prompt


def load_and_preprocess_data(dataset_path: str, tokenizer, max_length: int = 2048):
    """加载和预处理数据"""
    print(f"加载数据集: {dataset_path}")
    data = load_jsonl(dataset_path)
    print(f"共加载 {len(data)} 条数据")

    def process_function(examples):
        """处理数据"""
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i] if examples["instruction"][i] else ""
            input_text = examples["input"][i] if examples["input"][i] else ""

            # 使用教师模型的输出作为目标，优先使用 teacher_output
            if "teacher_output" in examples:
                output = examples["teacher_output"][i] if i < len(
                    examples["teacher_output"]) else ""
            elif "output" in examples:
                output = examples["output"][i] if i < len(
                    examples["output"]) else ""
            else:
                output = ""

            prompt = format_prompt(instruction, input_text, output)
            texts.append(prompt)

        # Tokenize - 使用相同的参数确保一致性
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=True  # 明确指定
        )

        # 设置 labels：只对 assistant 回复部分计算损失
        labels = []
        assistant_marker = "<|im_start|>assistant\n"

        for text, input_ids in zip(texts, tokenized["input_ids"]):
            # 确保 input_ids 是列表格式
            if not isinstance(input_ids, list):
                input_ids_list = list(input_ids)
            else:
                input_ids_list = list(input_ids)
            
            # 初始化 labels，复制 input_ids
            label = input_ids_list.copy()

            # 找到 assistant 标记在文本中的位置
            assistant_pos = text.find(assistant_marker)

            if assistant_pos != -1:
                # 使用与上面相同的 tokenizer 参数来 tokenize assistant 之前的部分
                before_text = text[:assistant_pos + len(assistant_marker)]
                # 使用 tokenizer 的 __call__ 方法确保一致性
                before_tokenized = tokenizer(
                    before_text,
                    add_special_tokens=True,
                    return_tensors=None
                )
                
                # 正确处理 input_ids - 确保是列表格式
                before_input_ids = before_tokenized["input_ids"]
                if before_input_ids and len(before_input_ids) > 0:
                    before_ids = before_input_ids[0]
                    # 确保 before_ids 是列表
                    if not isinstance(before_ids, list):
                        before_ids = [before_ids] if isinstance(before_ids, int) else list(before_ids)
                else:
                    before_ids = []
                
                # 计算 assistant 回复开始的索引
                start_idx = len(before_ids)
                
                # 确保 start_idx 在有效范围内
                if start_idx > len(input_ids_list):
                    start_idx = len(input_ids_list)
                
                # 将 prompt 部分（assistant 之前）设为 -100（忽略）
                for i in range(start_idx):
                    if i < len(label):
                        label[i] = -100
            # 如果没有 assistant 标记，对整个序列计算损失（不修改 label）

            # 确保 label 是整数列表
            label = [int(x) for x in label]
            labels.append(label)

        tokenized["labels"] = labels

        return tokenized

    # 转换为 Dataset
    dataset_dict = {
        "instruction": [item.get("instruction", "") for item in data],
        "input": [item.get("input", "") for item in data],
        "output": [item.get("output", "") for item in data],
    }

    # 添加 teacher_output 如果存在
    if data and "teacher_output" in data[0]:
        dataset_dict["teacher_output"] = [
            item.get("teacher_output", "") for item in data]

    dataset = Dataset.from_dict(dataset_dict)

    # 处理数据
    processed_dataset = dataset.map(
        process_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="处理数据"
    )

    return processed_dataset


def train():
    """主训练函数"""
    # 加载配置
    student_config = StudentConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig()

    # 检查数据文件
    if not os.path.exists(training_config.dataset_path):
        raise FileNotFoundError(
            f"数据文件不存在: {training_config.dataset_path}\n"
            f"请先运行 python prepare_data.py 准备数据"
        )

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(student_config)

    # 设置 QLoRA
    model = setup_lora(model, lora_config)

    # 加载数据
    train_dataset = load_and_preprocess_data(
        training_config.dataset_path,
        tokenizer,
        training_config.max_seq_length
    )

    # 如果有验证集，加载验证集
    eval_dataset = None
    if training_config.eval_dataset_path and os.path.exists(training_config.eval_dataset_path):
        eval_dataset = load_and_preprocess_data(
            training_config.eval_dataset_path,
            tokenizer,
            training_config.max_seq_length
        )

    # 数据整理器 - 使用支持 padding 的整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    # 训练参数
    # 如果没有验证集，不能使用 load_best_model_at_end
    load_best_model = training_config.load_best_model_at_end and eval_dataset is not None
    
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=load_best_model,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        optim=training_config.optim,
        lr_scheduler_type=training_config.lr_scheduler_type,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        logging_dir=f"{training_config.output_dir}/logs",
        report_to="none",
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 保存最终模型
    print(f"保存模型到 {training_config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_config.output_dir)

    print("训练完成！")


if __name__ == "__main__":
    train()
