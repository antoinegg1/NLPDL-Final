import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_from_disk
import json
from tqdm import tqdm

def main():
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        raise ValueError("CUDA不可用，请检查您的GPU设置。")
    SYSTEM_PROMPT = """Convert casual text to formal text: """
    
    # 获取所有可用的GPU设备
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    
    if num_gpus < 1:
        raise ValueError("至少需要一个GPU。")
    print(f"检测到 {num_gpus} 个GPU。")
    
    # 加载预训练的T5模型和分词器
    model_name = "/mnt/file2/changye/model/t5-formal-finetuned"  # 您可以根据需要选择不同的T5模型
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 将模型移动到GPU，并使用DataParallel包装以利用多GPU
    model = model.to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print("使用DataParallel包装模型。")
    
    # 加载数据集
    dataset = load_from_disk("/mnt/file2/changye/dataset/casual_formal_pair_ACL40k/test")
    
    # 定义预处理函数
    def preprocess_function(example):
        # 将系统提示与输入文本连接
        input_text = SYSTEM_PROMPT + example["casual_text"]
        # 使用tokenizer进行编码，不返回张量，而是返回列表
        model_inputs = tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=512
        )
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"]
        }
    
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
    
    # 设置DataLoader参数
    batch_size = 8  # 根据GPU内存调整批量大小
    
    # 准备存储结果
    results = []
    
    # 设置模型为评估模式
    model.eval()
    
    # 禁用梯度计算以提高推理速度
    with torch.no_grad():
        # 使用tqdm显示进度条
        for i in tqdm(range(0, len(tokenized_dataset), batch_size), desc="推理中"):
            # 获取一个批次的数据
            batch = tokenized_dataset[i:i + batch_size]
            
            # 将input_ids和attention_mask转换为张量并移动到GPU
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
            
            # 生成输出
            if num_gpus > 1:
                # 如果使用了DataParallel，需要通过module访问generate方法
                outputs = model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=50
                )
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=50
                )
            
            # 解码生成的文本
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 将结果添加到列表中
            for input_ids_example, output_text in zip(batch["input_ids"], decoded_outputs):
                input_text = tokenizer.decode(input_ids_example, skip_special_tokens=True)
                result = {
                    "casual_text": input_text.replace(SYSTEM_PROMPT, "").strip(),
                    "formal_text": output_text.strip(),
                }
                results.append(result)
    
    # 将结果保存到JSON文件
    output_file = "t5_inference_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"推理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
