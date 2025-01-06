import argparse
import os
import logging

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from model import load_model_and_tokenizer
from supervised_dataset import prepare_dataset
from datasets import load_from_disk
import wandb 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-like models with DeepSpeed & Weights & Biases.")

    # ===== 数据相关参数 =====
    parser.add_argument("--train_dataset", type=str, default="train.json", help="Path to the train dataset in JSON format.")
    parser.add_argument("--val_dataset", type=str, default="val.json", help="Path to the validation dataset in JSON format.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")

    # ===== 训练超参数 =====
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Where to store the final model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size *per device*.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for lr scheduler.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")

    # ===== 模型相关参数 =====
    parser.add_argument("--model_path", type=str, default="gpt2", help="Model name or path, e.g. 'gpt2', 't5-small' etc.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 if supported.")

    # ===== DeepSpeed & W&B =====
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="Path to DeepSpeed config file.")
    parser.add_argument("--wandb_project", type=str, default="my-project", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default="my-run", help="Weights & Biases run name.")

    return parser.parse_args()


def main():
    args = parse_args()

    # ===== Step 0: 初始化 wandb（可在此处或 Trainer 中自动启动） =====
    # 1) 确保您已经在命令行执行过 `wandb login`，或使用环境变量 WANDB_API_KEY
    # 2) 设置 wandb 项目名称、运行名称
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # ===== Step 1: 加载模型和分词器 =====
    model, tokenizer, model_type = load_model_and_tokenizer(args.model_path, args.local_rank)
    logger.info(f"Loaded model_type: {model_type}")

    # ===== Step 2: 加载原始数据并转换为 DatasetDict =====
    train_dataset = load_from_disk(args.train_dataset)
    eval_dataset = load_from_disk(args.val_dataset)
    logger.info("Dataset loaded successfully.")

    # ===== Step 3: 预处理数据 =====
    tokenized_train_dataset = prepare_dataset(train_dataset, tokenizer, args.max_length, model_type)
    tokenized_eval_dataset = prepare_dataset(eval_dataset, tokenizer, args.max_length, model_type)
    logger.info("Dataset tokenized successfully.")

    # ===== Step 4: 构建 DataCollator =====
    if "t5" in model_type.lower():
        # T5 一般使用 Seq2Seq 的 DataCollator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8 if args.fp16 else None
        )
    else:
        # GPT 类模型通常使用语言建模的 DataCollator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # GPT 类是自回归语言模型，不用 Masked LM
        )

    # ===== Step 5: 设置 TrainingArguments =====
    # 关键：将 deepspeed 的配置文件路径传给 training_args.deepspeed
    # 同时指定 report_to="wandb"，这样训练过程会自动上报到 wandb
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="steps",       
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        fp16=args.fp16,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        # =============================
        # DeepSpeed & W&B 关键参数
        # =============================
        deepspeed=args.deepspeed_config,  # DeepSpeed 配置文件路径
        report_to=["wandb"],              # 把日志上报到 wandb
        run_name=args.wandb_run_name      # wandb 中显示的 run 名称
    )

    # ===== Step 6: 初始化 Trainer =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ===== Step 7: 开始训练 =====
    logger.info("Starting training with DeepSpeed & W&B logging...")
    train_result = trainer.train()

    # ===== Step 8: 保存最终模型和分词器 =====
    logger.info("Training finished. Saving model and tokenizer.")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ===== Step 9: 结束并退出 wandb 运行 =====
    wandb.finish()

    logger.info("All done. Model saved at: %s", args.output_dir)


if __name__ == "__main__":
    main()
