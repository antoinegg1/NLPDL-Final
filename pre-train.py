from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_scheduler
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import torch

# 自定义 Dataset
class AcademicDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        """
        :param data: 输入的数据集 (texts 是论文段落列表)
        :param tokenizer: 用于处理文本的 GPT-2 分词器
        :param max_length: 最大输入长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取论文段落列表
        paragraphs = self.data[idx]['texts']
        # 合并段落为单一文本
        text = " ".join(paragraphs)
        # 分词和转换为 token IDs
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
            "labels": tokenized["input_ids"].squeeze(0)  # 自回归任务目标
        }

# 加载 GPT-2 分词器和模型
model_path="/mnt/file2/changye/model/gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
data_path="/mnt/file2/changye/dataset/ACL_clear/train"
data=load_from_disk(data_path)
# 配置数据集
max_length = 512
dataset = AcademicDataset(
    data=data,
    tokenizer=tokenizer,
    max_length=max_length
)

# 创建 DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(dataloader) * 3  # 假设训练 3 个 epoch
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 训练循环
epochs = 3
model.train()

for epoch in range(epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# 保存微调模型
model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")

print("Model fine-tuning complete and saved!")
