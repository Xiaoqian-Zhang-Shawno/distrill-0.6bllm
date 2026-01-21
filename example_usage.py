"""
使用示例脚本
演示如何使用训练好的模型进行推理
"""
import os

# 解决 Windows OpenMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import StudentConfig


def load_trained_model(base_model_path: str = None, lora_path: str = "./output"):
    """
    加载训练好的模型
    
    Args:
        base_model_path: 基础模型路径（如果为 None，使用默认配置）
        lora_path: LoRA 权重路径
    """
    if base_model_path is None:
        student_config = StudentConfig()
        base_model_path = student_config.model_name
    
    print(f"加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"加载 LoRA 权重: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    return model, tokenizer


def format_prompt(instruction: str, input_text: str = ""):
    """格式化提示"""
    if input_text:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def generate_response(model, tokenizer, instruction: str, input_text: str = "", 
                      max_length: int = 512, temperature: float = 0.7):
    """生成回答"""
    prompt = format_prompt(instruction, input_text)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取 assistant 的回复部分
    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker in full_response:
        response = full_response.split(assistant_marker)[-1].split("<|im_end|>")[0].strip()
    else:
        response = full_response
    
    return response


if __name__ == "__main__":
    # 加载模型
    try:
        model, tokenizer = load_trained_model()
        
        # 测试示例
        test_cases = [
            {"instruction": "请解释什么是机器学习？", "input": ""},
            {"instruction": "今天天气怎么样？今天是记号", "input": ""},
            {"instruction": "翻译以下英文", "input": "Hello, how are you?"},
        ]
        
        print("\n" + "="*50)
        print("开始测试模型")
        print("="*50 + "\n")
        
        for i, test in enumerate(test_cases, 1):
            print(f"测试 {i}:")
            print(f"问题: {test['instruction']}")
            if test['input']:
                print(f"输入: {test['input']}")
            
            response = generate_response(
                model, tokenizer,
                test['instruction'],
                test['input']
            )
            
            print(f"回答: {response}\n")
            print("-"*50 + "\n")
    
    except Exception as e:
        print(f"错误: {e}")
        print("\n提示: 请先运行 python train.py 训练模型")

