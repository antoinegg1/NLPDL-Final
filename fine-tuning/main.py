# main.py
'''
torchrun --nproc_per_node=8 main.py \
    --train_file /mnt/file2/changye/dataset/casual_formal_pair_ACL40k/train \
    --val_file /mnt/file2/changye/dataset/casual_formal_pair_ACL40k/val \
    --output_dir /mnt/file2/changye/model/gpt2-formal-finetuned \
    --logging_dir ./logs \
    --num_epochs 3 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --logging_steps 100 \
    --max_length 512 \
    --dataloader_num_workers 60 \
    --gradient_accumulation_steps 1 \
    --wandb_project academic_finetuning \
'''
import argparse
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
from trainer import initialize_trainer
import wandb
from datasets import load_from_disk

# 从 supervised_dataset.py 导入 SYSTEM_PROMPT
from supervised_dataset import SYSTEM_PROMPT,prepare_dataset

def convert_casual_to_formal(casual_text: str, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, max_length: int = 512) -> str:
    """
    使用微调后的模型将非正式文本转换为正式文本。

    Args:
        casual_text (str): 非正式文本。
        model (GPT2LMHeadModel): 微调后的 GPT-2 模型。
        tokenizer (GPT2Tokenizer): GPT-2 分词器。
        max_length (int): 生成文本的最大长度。

    Returns:
        str: 生成的正式文本。
    """
    input_text = f"{SYSTEM_PROMPT}{casual_text} Formal: "
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

    # 生成正式文本
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    # 解码生成的文本
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # 提取正式文本部分
    formal_text = output_text.split("Formal:")[-1].strip()
    return formal_text

def main(args):
    # 创建输出目录和日志目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    # 初始化 wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        job_type="fine-tuning"
    )

    # 设置随机种子以确保结果可复现
    set_seed(42)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据集
    # 使用 Hugging Face datasets 的 load_from_disk 加载预先保存的数据集
    train_dataset = load_from_disk(args.train_file)
    eval_dataset = load_from_disk(args.val_file)
    print(f"训练集加载完成，共有 {len(train_dataset)} 个样本。")
    print(f"验证集加载完成，共有 {len(eval_dataset)} 个样本。")

    # 初始化分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 添加填充标记（如果尚未添加）
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("添加了填充标记 '[PAD]' 到分词器。")

    # 加载数据并进行预处理
    tokenized_train = prepare_dataset(train_dataset, tokenizer, max_length=args.max_length)
    tokenized_eval = prepare_dataset(eval_dataset, tokenizer, max_length=args.max_length)
    print("数据预处理完成。")

    # 合并预处理后的数据集
    tokenized_dataset = {"train": tokenized_train, "validation": tokenized_eval}

    # 加载预训练的 GPT-2 模型
    model = GPT2LMHeadModel.from_pretrained('/mnt/file2/changye/model/fine_tuned_gpt2')
    model.resize_token_embeddings(len(tokenizer))  # 调整词汇表大小
    model.to(device)
    print("模型加载并调整词汇表完成。")

    # 初始化 Trainer
    trainer = initialize_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        logging_steps=args.logging_steps,
        max_length=args.max_length,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # 将 Trainer 配置为使用 wandb
    trainer.args.report_to = ['wandb']

    # 开始训练
    trainer.train()

    # 保存模型和分词器
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型和分词器已保存到 {args.output_dir}")

    # 结束 wandb 运行
    wandb.finish()

    # 推理示例
    def inference_example():
        example_casual = "Hey, what's up? Can you help me with this?"
        formal = convert_casual_to_formal(example_casual, model, tokenizer, max_length=args.max_length)
        print(f"非正式文本: {example_casual}")
        print(f"正式文本: {formal}")

    inference_example()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for Casual to Formal Text Conversion")

    # 数据路径
    parser.add_argument('--train_file', type=str, required=True, help='训练数据集的目录路径（使用 Hugging Face datasets 的 load_from_disk）')
    parser.add_argument('--val_file', type=str, required=True, help='验证数据集的目录路径（使用 Hugging Face datasets 的 load_from_disk）')

    # 输出路径
    parser.add_argument('--output_dir', type=str, default='./gpt2-formal-finetuned', help='模型和分词器的输出目录')
    parser.add_argument('--logging_dir', type=str, default='./logs', help='日志目录')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--train_batch_size', type=int, default=8, help='训练批次大小（每个 GPU）')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='验证批次大小（每个 GPU）')
    parser.add_argument('--logging_steps', type=int, default=100, help='日志记录步数')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')

    # 数据加载参数
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='数据加载的工作线程数')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')

    # wandb 参数
    parser.add_argument('--wandb_project', type=str, required=True, help='wandb 项目名称')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb 实体名称（团队或用户）')

    args = parser.parse_args()

    main(args)
