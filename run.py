"""
快速启动脚本
自动执行完整的数据准备和训练流程
"""
from config import TeacherConfig, TrainingConfig
from train import train
from prepare_data import prepare_training_data
import os
import sys

# 解决 Windows OpenMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    """主函数"""
    print("="*60)
    print("知识蒸馏训练流程")
    print("="*60)

    # 检查 API Key
    teacher_config = TeacherConfig()
    if not teacher_config.api_key:
        print("\n错误: 未设置 DEEPSEEK_API_KEY")
        print("请设置环境变量或在 config.py 中配置 API Key")
        print("\nWindows: set DEEPSEEK_API_KEY=your_key")
        print("Linux/Mac: export DEEPSEEK_API_KEY=your_key")
        sys.exit(1)

    # 配置路径
    training_config = TrainingConfig()
    raw_data_path = "./data/raw_data.jsonl"
    train_data_path = training_config.dataset_path
    eval_data_path = training_config.eval_dataset_path or "./data/eval_dataset.jsonl"

    # 步骤 1: 检查原始数据
    print("\n步骤 1: 检查数据文件...")
    if not os.path.exists(raw_data_path):
        print(f"警告: 原始数据文件不存在: {raw_data_path}")
        print("请创建数据文件，格式为 JSONL，每行一个 JSON 对象：")
        print('{"instruction": "请解释什么是机器学习？", "input": ""}')
        print('{"instruction": "翻译以下英文", "input": "Hello, how are you?"}')

        # 创建示例数据
        create_example = input("\n是否创建示例数据文件？(y/n): ").strip().lower()
        if create_example == 'y':
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            import json
            example_data = [
                {"instruction": "请解释什么是机器学习？", "input": ""},
                {"instruction": "写一首关于春天的诗", "input": ""},
                {"instruction": "翻译以下英文", "input": "Hello, how are you?"},
                {"instruction": "什么是深度学习？", "input": ""},
                {"instruction": "解释一下 Python 的列表推导式", "input": ""},
            ]
            with open(raw_data_path, "w", encoding="utf-8") as f:
                for item in example_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"已创建示例数据文件: {raw_data_path}")
        else:
            sys.exit(1)

    # 步骤 2: 生成教师模型回答（训练集和验证集）
    print("\n步骤 2: 使用教师模型生成回答...")
    print("这可能需要一些时间，取决于数据量...")
    print(f"验证集比例: {training_config.eval_split_ratio:.1%}")

    try:
        train_data, eval_data = prepare_training_data(
            input_file=raw_data_path,
            output_file=train_data_path,
            teacher_config=teacher_config,
            eval_output_file=eval_data_path,
            eval_ratio=training_config.eval_split_ratio,
            eval_seed=training_config.eval_split_seed
        )

        # 更新训练配置中的验证集路径
        training_config.eval_dataset_path = eval_data_path

        print(f"\n✓ 训练集: {len(train_data)} 条，已保存到 {train_data_path}")
        if eval_data:
            print(f"✓ 验证集: {len(eval_data)} 条，已保存到 {eval_data_path}")
    except Exception as e:
        print(f"错误: 生成教师模型回答失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 步骤 3: 开始训练
    print("\n步骤 3: 开始训练学生模型...")
    print("这可能需要较长时间，请耐心等待...")

    try:
        train()
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        print(f"\n模型已保存到: {training_config.output_dir}")
        print("\n使用训练好的模型:")
        print("python example_usage.py")
    except Exception as e:
        print(f"错误: 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
