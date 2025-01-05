import os
import wandb
import argparse
import deepspeed
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_scheduler,
    AutoTokenizer,
    AutoModelForCausalLM
)
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import torch
from torch.optim import AdamW
from tqdm import tqdm

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune models for academic style text generation")
    parser.add_argument("--model_type", type=str, default="qwen", help="Model type to use (gpt2, t5, etc.)")
    parser.add_argument("--model_path", type=str, default="/mnt/file2/changye/model/NLP/Qwen2.5-1.5B-Instruct", help="Path to the pre-trained model")
    parser.add_argument("--train_data_path", type=str, default="/mnt/file2/changye/dataset/NLP/ACL_clear/train", help="Path to the training dataset")
    parser.add_argument("--val_data_path", type=str, default="/mnt/file2/changye/dataset/NLP/ACL_clear/val", help="Path to the validation dataset")
    parser.add_argument("--output_path", type=str, default="/mnt/file2/changye/model/qwen-formal-trained", help="Path to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save_every", type=int, default=500, help="Steps interval for saving checkpoints")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="Path to DeepSpeed config file")
    
    # 解析已知参数，忽略未知参数（如 --local_rank）
    args, unknown = parser.parse_known_args()
    return args

# Dataset Class
class AcademicDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paragraphs = self.data[idx]['texts']
        text = " ".join(paragraphs)
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0)  # 自回归任务中 labels 通常与 input_ids 相同
        }

def collate_fn(batch):
    # 过滤掉异常数据
    batch = [item for item in batch if item["input_ids"].size(0) > 0]
    if len(batch) == 0:
        return None
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }

# Function to load model
def load_model(model_type, model_path):
    if model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_path)
    elif model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

# Function to load datasets
def load_datasets(train_data_path, val_data_path, tokenizer, max_length):
    train_data = load_from_disk(train_data_path)
    val_data = load_from_disk(val_data_path)

    train_dataset = AcademicDataset(train_data, tokenizer, max_length)
    val_dataset = AcademicDataset(val_data, tokenizer, max_length)

    return train_dataset, val_dataset

# Main Function
def main():
    args = parse_args()

    # 1. 设置 wandb
    wandb.init(project="academic-style-text-gen", name=f"{args.model_type}-finetune")

    # 2. 准备模型与分词器
    model, tokenizer = load_model(args.model_type, args.model_path)

    # 3. 准备 DataLoader
    train_dataset, val_dataset = load_datasets(args.train_data_path, args.val_data_path, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # DeepSpeed 会按 GPU 数量拆分
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4
    )

    # 4. 设置优化器（DeepSpeed 会自动处理）
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # 5. 初始化 DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
    )

    # 6. 计算总训练步数
    # num_training_steps = len(train_loader) * args.epochs
    # lr_scheduler = get_scheduler(
    #     name="cosine",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps
    # )

    global_step = 0
    for epoch in range(args.epochs):
        model_engine.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = 0.0

        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue
            # 将 batch 搬到当前进程所用的 GPU
            batch = {k: v.to(model_engine.local_rank, non_blocking=True) for k, v in batch.items()}

            # 前向传播
            outputs = model_engine(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss

            # DeepSpeed 后向传播
            model_engine.backward(loss)
            # DeepSpeed 更新
            model_engine.step()

            # 记录与日志
            step_loss = loss.item()
            train_loss += step_loss
            global_step += 1
            wandb.log({"train_step_loss": step_loss, "global_step": global_step})

            # 打印进度
            progress_bar.set_postfix({"loss": step_loss})

            # 保存中间检查点
            if global_step % args.save_every == 0:
                # 使用 DeepSpeed 的方式保存 checkpoint
                ckpt_save_path = model_engine.save_checkpoint(
                    args.output_path,
                    tag=f"checkpoint_{global_step}"
                )
                print(f"Saved DeepSpeed checkpoint to {ckpt_save_path}")

        # ===== 每个 epoch 结束后做验证 =====
        model_engine.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch + 1}/{args.epochs}"):
                if batch is None:
                    continue
                batch = {k: v.to(model_engine.local_rank, non_blocking=True) for k, v in batch.items()}
                outputs = model_engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                val_loss += outputs.loss.item()

        val_loss /= max(1, len(val_loader))
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss})
        print(f"Epoch {epoch + 1}: Validation Loss = {val_loss:.4f}")

    # 7. 训练完成后保存最终模型（Hugging Face 格式）
    final_model_path = os.path.join(args.output_path, f"fine_tuned_{args.model_type}")
    if not os.path.exists(final_model_path):
        os.makedirs(final_model_path, exist_ok=True)

    # 保存权重和分词器
    model_engine.module.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    wandb.finish()
    print("Model fine-tuning complete and saved with DeepSpeed!")

if __name__ == "__main__":
    main()
