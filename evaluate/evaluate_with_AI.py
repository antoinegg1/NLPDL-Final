import json
import time
import requests
from datasets import load_dataset, load_from_disk, concatenate_datasets
import tqdm
from openai import OpenAI
import re
import random
import argparse
import torch
from datasets import Dataset
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import os
    # 设置系统提示语
system_prompt = """
    Evaluate the quality of the following style-transferred sentence:
    
    Original Sentence: "{source_sentence}"
    Transferred Sentence: "{transferred_sentence}"
    Target Style: "{target_style}"
    
    Evaluate using these metrics:
    1. Style Transfer Strength: Predict how likely the transferred sentence matches the target style (score from 0 to 1).
    2. Content Preservation: Evaluate how much original content is preserved in the transferred sentence (score from 0 to 1).
    3. Fluency: Rate the fluency of the transferred sentence (score from 0 to 1).
    
    The Output must only be in the following JSON format:
    {{
        "Style Transfer Strength": [score],
        "Content Preservation": [score],
        "Fluency": [score]
    }}
    """
def get_deepseek_response(API_KEY, API_URL, prompt):
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {e}")
        return None
    
def process_example(example, API_KEY, API_URL):
    prompt = system_prompt.format(source_sentence=example['casual_text'], transferred_sentence=example['Model_formal_text'], target_style=example['formal_text'])
    generated_answer = get_deepseek_response(API_KEY, API_URL, prompt)
    return {
        "directory": example['directory'],
        "filename": example['filename'],
        "formal_text": example['formal_text'],
        "casual_text": example['casual_text'],
        "Model_formal_text": example['Model_formal_text'],
        "evaluation": generated_answer
    }

def save_chunk(results, save_path, chunk_index):
    if not results:
        return
    dataset_out = Dataset.from_dict({
        "evaluation": [item["evaluation"] for item in results],
        "directory": [item["directory"] for item in results],
        "filename": [item["filename"] for item in results],
        "formal_text": [item["formal_text"] for item in results],
        "casual_text": [item["casual_text"] for item in results],
        "Model_formal_text": [item["Model_formal_text"] for item in results],
    })
    # 构造分段文件名
    chunk_save_path = f"{save_path}_part{chunk_index}"
    dataset_out.save_to_disk(chunk_save_path)
    print(f"已保存第 {chunk_index} 部分到 {chunk_save_path}")

def run_experiment(save_path, max_workers=10, chunk_size=10000,dataset_path="/mnt/file2/changye/NLPFINAL/result/mistral_formal_text_result.json", sample_size=1000):
    # 请替换为实际的API端点与API密钥
    API_URL = "https://api.deepseek.com"
    API_KEY = "sk-b152415699674f74a0046c553238e2f3"  # 修改为你的API Key
    if "json" in dataset_path:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
    else:
        dataset = load_from_disk(dataset_path)

    sample_dataset = random.sample(dataset, sample_size)


    results = []
    total_count = len(sample_dataset)
    chunk_index = 1  # 分段文件索引

    # 确保保存路径目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 使用 ThreadPoolExecutor 进行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建一个部分函数，固定 API_KEY, API_URL, system_prompt
        partial_process = partial(process_example, API_KEY=API_KEY, API_URL=API_URL)
        
        # 提交所有任务
        futures = {executor.submit(partial_process, example): example for example in sample_dataset}

        # 使用 tqdm 显示进度条
        for future in tqdm.tqdm(as_completed(futures), total=total_count, desc="Running experiment"):
            result = future.result()
            if result["evaluation"] is not None:
                results.append(result)
            
            # 检查是否达到分段保存的大小
            if len(results) >= chunk_size:
                save_chunk(results, save_path, chunk_index)
                chunk_index += 1
                results = []  # 清空当前结果列表以准备下一段

    # 保存剩余不足一个分段大小的结果
    if results:
        save_chunk(results, save_path, chunk_index)

    print(f"所有实验结果已保存到 {save_path} 的各个分段文件中。")

def main():
    parser = argparse.ArgumentParser(description="Run GSM8K experiment with different settings.")
    parser.add_argument("--save_path", type=str, default="/mnt/file2/changye/NLPFINAL/eval_result/Qwen2.5-1.5B-Instruct-formal-trained_paragraph")
    parser.add_argument("--max_workers", type=int, default=60, help="并行工作的最大线程数")
    parser.add_argument("--chunk_size", type=int, default=10000, help="每个分段保存的样本数")
    parser.add_argument("--dataset_path", type=str, default="/mnt/file2/changye/NLPFINAL/Generate_result/Qwen2.5-1.5B-Instruct-formal-trained_paragraph.json", help="Path to the dataset to load using load_from_disk")
    parser.add_argument("--sample_size", type=int, default=300, help="样本数量")
    args = parser.parse_args()

    run_experiment(args.save_path, max_workers=args.max_workers, chunk_size=args.chunk_size, dataset_path=args.dataset_path, sample_size=args.sample_size)

if __name__ == "__main__":
    main()
