import os
import json
from datasets import Dataset

def extract_texts_from_json(json_file_path):
    """从单个 JSON 文件中提取 body_text 的 text 字段列表"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    body_text_list = data.get("pdf_parse", {}).get("body_text", [])
    texts = [entry.get("text", "") for entry in body_text_list]
    return texts

def process_folder(input_folder, dataset_output):
    # 用于存储数据的列表
    # 每个元素将是一个字典，如 {"filename": "xxx.json", "texts": ["...","..."]}
    records = []

    # 遍历文件夹内的所有文件
    for sub_dir in os.listdir(input_folder):
        sub_dir_path = os.path.join(input_folder, sub_dir)
        
        # 判断是否为文件夹
        if os.path.isdir(sub_dir_path):
            # 遍历该文件夹下的所有 JSON 文件
            for filename in os.listdir(sub_dir_path):
                if filename.endswith(".json"):
                    json_path = os.path.join(sub_dir_path, filename)
                    texts = extract_texts_from_json(json_path)

                    record = {
                        "directory": sub_dir,
                        "filename": filename,
                        "texts": texts
                    }
                    records.append(record)

    # 将数据转换为 HuggingFace Dataset
    dataset = Dataset.from_list(records)
    
    # 将 Dataset 保存到指定目录
    dataset.save_to_disk(dataset_output)
    print("数据集已保存至:", dataset_output)


if __name__ == "__main__":
    input_folder = "/mnt/file2/changye/dataset/ACL-OCL/Base_JSON/prefixE/json/"    # 替换为你的 JSON 文件所在的文件夹路径
    dataset_output = "/mnt/file2/changye/dataset/ACL_clear/E" # 替换为你的输出数据集路径
    process_folder(input_folder, dataset_output)
