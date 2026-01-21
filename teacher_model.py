"""
教师模型 API 调用模块
使用 DeepSeek API 生成回答
"""
import os
import time
import json
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm
from config import TeacherConfig


class TeacherModel:
    """教师模型类，通过 DeepSeek API 调用"""
    
    def __init__(self, config: TeacherConfig):
        self.config = config
        if not config.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量或在 config.py 中配置")
        
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成单个回答
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
        
        Returns:
            生成的回答
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API 调用错误: {e}")
            return ""
    
    def generate_batch(self, prompts: List[str], system_prompt: Optional[str] = None, 
                       delay: float = 0.1) -> List[str]:
        """
        批量生成回答
        
        Args:
            prompts: 提示列表
            system_prompt: 系统提示（可选）
            delay: 每次调用之间的延迟（秒），避免 API 限流
        
        Returns:
            生成的回答列表
        """
        results = []
        for prompt in tqdm(prompts, desc="教师模型生成中"):
            result = self.generate(prompt, system_prompt)
            results.append(result)
            time.sleep(delay)  # 避免 API 限流
        
        return results
    
    def generate_dataset(self, input_data: List[Dict], output_path: str):
        """
        为数据集生成教师模型的回答
        
        Args:
            input_data: 输入数据列表，每个元素包含 "instruction" 或 "input" 字段
            output_path: 输出文件路径（JSONL 格式）
        """
        print(f"开始为 {len(input_data)} 条数据生成教师模型回答...")
        
        generated_data = []
        for i, item in enumerate(tqdm(input_data, desc="生成数据集")):
            # 构建提示
            if "instruction" in item:
                prompt = item["instruction"]
                if "input" in item and item["input"]:
                    prompt = f"{item['instruction']}\n\n{item['input']}"
            elif "input" in item:
                prompt = item["input"]
            else:
                print(f"警告: 第 {i+1} 条数据缺少 instruction 或 input 字段，跳过")
                continue
            
            # 生成回答
            response = self.generate(prompt)
            
            if response:
                new_item = {
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": response,
                    "teacher_output": response  # 保存教师模型的输出
                }
                generated_data.append(new_item)
            else:
                print(f"警告: 第 {i+1} 条数据生成失败，跳过")
            
            # 避免 API 限流
            time.sleep(0.1)
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in generated_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"已生成 {len(generated_data)} 条数据，保存到 {output_path}")


if __name__ == "__main__":
    # 测试代码
    config = TeacherConfig()
    teacher = TeacherModel(config)
    
    test_prompt = "请解释什么是机器学习？"
    response = teacher.generate(test_prompt)
    print(f"提示: {test_prompt}")
    print(f"回答: {response}")

