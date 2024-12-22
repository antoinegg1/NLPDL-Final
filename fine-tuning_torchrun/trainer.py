# trainer.py

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,Seq2SeqTrainer
from datasets import DatasetDict
import torch

def initialize_trainer(
    model,
    tokenizer,
    train_dataset: DatasetDict,
    eval_dataset: DatasetDict,
    output_dir: str,
    logging_dir: str,
    num_epochs: int = 3,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    logging_steps: int = 100,
    max_length: int = 512,
    dataloader_num_workers: int = 4,
    gradient_accumulation_steps: int = 1,
    model_type: str = 'gpt2'
) -> Trainer:
    """
    初始化 Hugging Face 的 Trainer。

    Args:
        model: 预训练的模型。
        tokenizer: 对应的分词器。
        train_dataset (DatasetDict): 训练数据集。
        eval_dataset (DatasetDict): 验证数据集。
        output_dir (str): 模型和分词器的输出目录。
        logging_dir (str): 日志目录。
        num_epochs (int): 训练轮数。
        train_batch_size (int): 训练批次大小（每个 GPU）。
        eval_batch_size (int): 验证批次大小（每个 GPU）。
        logging_steps (int): 每隔多少步记录一次日志。
        max_length (int): 最大序列长度。
        dataloader_num_workers (int): 数据加载的工作线程数。
        gradient_accumulation_steps (int): 梯度累积步数。
        model_type (str): 模型类型，'gpt2' 或 't5'。

    Returns:
        Trainer: 初始化好的 Trainer 对象。
    """
    if 'gpt2' in model_type.lower() or 'mistral' in model_type.lower():
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # GPT-2 是自回归模型，不使用掩码语言模型
        )
        training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,

        evaluation_strategy='steps',
        save_strategy='steps',
        save_steps=logging_steps,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        fp16=torch.cuda.is_available(),  # 启用 FP16 混合精度
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to='wandb',  # 报告到 wandb
        run_name='gpt2-model-finetuning',  # 可选：wandb run 名称
        logging_first_step=True,
        logging_strategy='steps',
        eval_steps=logging_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    elif 't5' in model_type.lower():
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            evaluation_strategy='steps',
            save_strategy='steps',
            save_steps=logging_steps,
            logging_dir=logging_dir,
            logging_steps=logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            fp16=torch.cuda.is_available(),  # 启用 FP16 混合精度
            dataloader_num_workers=dataloader_num_workers,
            gradient_accumulation_steps=gradient_accumulation_steps,
            report_to='wandb',  # 报告到 wandb
            run_name='t5-model-finetuning',  # 可选：wandb run 名称
            logging_first_step=True,
            learning_rate=3e-5,  # 调整学习率
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        return trainer
    else:
        raise ValueError("Unsupported model type. Please choose 'gpt2' or 't5'.")
    

    return trainer
