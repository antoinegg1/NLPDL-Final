# supervise_dataset.py

from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer
from typing import Dict

# 系统提示
SYSTEM_PROMPT = """You are a helpful assistant. Your task is to convert casual text into formal text without changing the original meaning or altering the order of sentences. Keep the tone formal and professional, using appropriate language while ensuring that the core ideas remain intact. Please take the following casual text and rewrite it in a more formal way:
Casual: """

def load_data(train_path: str, val_path: str) -> DatasetDict:
    """
    加载训练和验证数据集。

    Args:
        train_path (str): 训练数据的 JSON 文件路径。
        val_path (str): 验证数据的 JSON 文件路径。

    Returns:
        DatasetDict: 包含 'train' 和 'validation' 数据集。
    """
    train_data = Dataset.from_json(train_path)
    val_data = Dataset.from_json(val_path)
    return DatasetDict({
        'train': train_data,
        'validation': val_data
    })

def preprocess_function(examples: Dict[str, list], tokenizer: GPT2Tokenizer, max_length: int = 512) -> Dict[str, list]:
    """
    预处理函数：构建输入和目标文本，并进行分词。

    Args:
        examples (dict): 包含 'casual_text' 和 'formal_text' 的字典。
        tokenizer (GPT2Tokenizer): GPT-2 分词器。
        max_length (int): 最大序列长度。

    Returns:
        dict: 分词后的输入和标签。
    """
    # 构建输入序列：系统提示 + 非正式文本 + " Formal:"
    inputs = [f"{SYSTEM_PROMPT}{casual} Formal:" for casual in examples['casual_text']]
    targets = examples['formal_text']

    # 分词输入
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )

    # 分词目标文本
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )['input_ids']

    # 设置标签：系统提示和非正式文本部分的标签设为 -100，正式文本部分为实际 token ids
    # 找到 " Formal:" 的 token 数量
    formal_prefix = " Formal:"
    formal_prefix_ids = tokenizer.encode(formal_prefix, add_special_tokens=False)
    prefix_length = len(formal_prefix_ids)

    processed_labels = []
    for input_ids, target_ids in zip(model_inputs['input_ids'], labels):
        # 设置前缀部分为 -100
        label = [-100] * (len(input_ids) - len(target_ids)) + target_ids
        # 如果长度不够，填充 -100
        if len(label) < max_length:
            label = label + [-100] * (max_length - len(label))
        else:
            label = label[:max_length]
        processed_labels.append(label)

    model_inputs['labels'] = processed_labels

    return model_inputs

def prepare_dataset(dataset: DatasetDict, tokenizer: GPT2Tokenizer, max_length: int = 512) -> DatasetDict:
    """
    应用预处理函数到整个数据集。

    Args:
        dataset (DatasetDict): 原始数据集。
        tokenizer (GPT2Tokenizer): GPT-2 分词器。
        max_length (int): 最大序列长度。

    Returns:
        DatasetDict: 预处理后的数据集。
    """
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length=max_length),
        batched=True,
    )
    return tokenized_dataset
