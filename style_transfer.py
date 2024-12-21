
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
import torch
from tqdm import tqdm

model_type = "t5"

if model_type == "gpt2":
    model_path="/mnt/file2/changye/model/fine_tuned_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
elif model_type == "t5":
    model_path="/mnt/file2/changye/model/fine_tuned_t5/fine_tuned_t5"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
        # 定义模型和优化器
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    # checkpoint = torch.load("/mnt/file2/changye/model/fine_tuned_t5/checkpoint_500.pt")  # 加载文件
    # model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数

def generate_text(prompt, max_length=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1,min_length=50, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

input_text = "summarize: In recent research, more attention has been paid" # input text
output_text = generate_text(input_text)
print(output_text)