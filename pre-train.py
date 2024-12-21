import os
import wandb
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel, get_scheduler
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import torch
from torch.optim import AdamW
from tqdm import tqdm

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune models for academic style text generation")
    parser.add_argument("--model_type", type=str, default="t5", choices=["gpt2", "t5"], help="Model type to use (gpt2 or t5)")
    parser.add_argument("--model_path", type=str, default="/mnt/file2/changye/model/t5-large", help="Path to the pre-trained model")
    parser.add_argument("--train_data_path", type=str, default="/mnt/file2/changye/dataset/ACL_clear/train", help="Path to the training dataset")
    parser.add_argument("--val_data_path", type=str, default="/mnt/file2/changye/dataset/ACL_clear/val", help="Path to the validation dataset")
    parser.add_argument("--output_path", type=str, default="/mnt/file2/changye/model/fine_tuned_t5", help="Path to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save_every", type=int, default=500, help="Steps interval for saving checkpoints")
    return parser.parse_args()

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
            "labels": tokenized["input_ids"].squeeze(0) if "input_ids" in tokenized else tokenized["input_ids"].squeeze(0)
        }

def collate_fn(batch):
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
        raise ValueError("Unsupported model type. Choose between 'gpt2' and 't5'.")
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

    # Initialize wandb
    wandb.init(project="academic-style-text-gen", name=f"{args.model_type}-finetune")

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_type, args.model_path)

    # Set up device and parallelism
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Load datasets
    train_dataset, val_dataset = load_datasets(args.train_data_path, args.val_data_path, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.epochs
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = 0
        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)


            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean()
            wandb.log({"step": step, "loss": loss})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({"loss": loss.item()})

            # Save checkpoint
            if global_step % args.save_every == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                }
                checkpoint_path = os.path.join(args.output_path, f"checkpoint_{global_step}.pt")
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint at step {global_step}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch + 1}/{args.epochs}"):
                if batch is None:
                    continue

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss.mean()


                val_loss += loss.item()

            val_loss /= len(val_loader)
            wandb.log({"epoch": epoch + 1, "val_loss": val_loss})
            print(f"Epoch {epoch + 1}: Validation Loss = {val_loss:.4f}")

    # Save final model
    final_model_path = os.path.join(args.output_path, f"fine_tuned_{args.model_type}")
    if isinstance(model, torch.nn.DataParallel):
        model.module.save_pretrained(final_model_path)
    else:
        model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    wandb.finish()
    print("Model fine-tuning complete and saved!")

if __name__ == "__main__":
    main()
