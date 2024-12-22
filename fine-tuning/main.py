# main.py
'''
torchrun --nproc_per_node=8 main.py \
    --train_file /mnt/file2/changye/dataset/casual_formal_pair_ACL40k/train \
    --val_file /mnt/file2/changye/dataset/casual_formal_pair_ACL40k/val \
    --output_dir /mnt/file2/changye/model/t5-formal-finetuned \
    --logging_dir ./logs \
    --num_epochs 3 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --logging_steps 100 \
    --max_length 512 \
    --dataloader_num_workers 60 \
    --gradient_accumulation_steps 1 \
    --wandb_project academic_finetuning \
    --model_name_or_path /mnt/file2/changye/model/fine_tuned_t5/fine_tuned_t5
'''
import argparse
import os
import torch
from transformers import set_seed
from trainer import initialize_trainer
from datasets import load_from_disk
import wandb
import torch.distributed as dist

# 从 supervised_dataset.py 导入 prepare_dataset
from supervised_dataset import prepare_dataset

# 导入模型加载函数
from model import load_model_and_tokenizer


def is_main_process():
    """
    判断当前进程是否为主进程（rank 0）。
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def main(args):
    # 初始化分布式训练环境
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        # For single process
        dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
        torch.cuda.set_device(args.local_rank)
    # 设置随机种子以确保结果可复现
    set_seed(42)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_main_process():
        print(f"使用设备: {device}")

    # 创建输出目录和日志目录（如果不存在）
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)

    # 初始化 wandb 仅在主进程
    if is_main_process():
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            job_type="fine-tuning"
        )

    # 加载数据集（使用 Hugging Face datasets 的 load_from_disk）
    train_dataset = load_from_disk(args.train_file)
    eval_dataset = load_from_disk(args.val_file)
    if is_main_process():
        print(f"训练集加载完成，共有 {len(train_dataset)} 个样本。")
        print(f"验证集加载完成，共有 {len(eval_dataset)} 个样本。")

    # 加载模型和分词器
    model, tokenizer, model_type = load_model_and_tokenizer(args.model_name_or_path, args.local_rank)

    
    # 加载数据并进行预处理
    tokenized_train = prepare_dataset(
        train_dataset,
        tokenizer,
        max_length=args.max_length,
        model_type=model_type
    )
    tokenized_eval = prepare_dataset(
        eval_dataset,
        tokenizer,
        max_length=args.max_length,
        model_type=model_type
    )
    if is_main_process():
        print("数据预处理完成。")

    # 合并预处理后的数据集
    tokenized_dataset = {"train": tokenized_train, "validation": tokenized_eval}

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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        model_type=model_type
    )

    # 将 Trainer 配置为使用 wandb
    trainer.args.report_to = ['wandb']

    # 开始训练
    trainer.train()

    # 保存模型和分词器（仅主进程）
    if is_main_process():
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"模型和分词器已保存到 {args.output_dir}")

    # 结束 wandb 运行（仅主进程）
    if is_main_process():
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 or T5 for Casual to Formal Text Conversion")

    # 数据路径
    parser.add_argument('--train_file', type=str, required=True, help='训练数据集的目录路径（使用 Hugging Face datasets 的 load_from_disk）')
    parser.add_argument('--val_file', type=str, required=True, help='验证数据集的目录路径（使用 Hugging Face datasets 的 load_from_disk）')

    # 输出路径
    parser.add_argument('--output_dir', type=str, default='./gpt2-formal-finetuned', help='模型和分词器的输出目录')
    parser.add_argument('--logging_dir', type=str, default='./logs', help='日志目录')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--train_batch_size', type=int, default=2, help='训练批次大小（每个 GPU）')
    parser.add_argument('--eval_batch_size', type=int, default=2, help='验证批次大小（每个 GPU）')
    parser.add_argument('--logging_steps', type=int, default=100, help='日志记录步数')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')

    # 数据加载参数
    parser.add_argument('--dataloader_num_workers', type=int, default=60, help='数据加载的工作线程数')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')

    # wandb 参数
    parser.add_argument('--wandb_project', type=str, required=True, help='wandb 项目名称')

    # 模型和系统提示
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help="模型名称或路径，例如 'gpt2' 或 't5-small'")

    # 分布式训练参数
    parser.add_argument('--local_rank', type=int, default=-1, help='本地 GPU 编号，用于分布式训练')

    args = parser.parse_args()

    main(args)
