# supervised_dataset.py

from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, T5Tokenizer
from typing import Dict

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

def preprocess_function(examples: Dict[str, list], tokenizer, max_length: int = 512, model_type: str = 'gpt2') -> Dict[str, list]:
    """
    预处理函数：构建输入和目标文本，并进行分词。

    Args:
        examples (dict): 包含 'casual_text' 和 'formal_text' 的字典。
        tokenizer: 分词器（GPT2Tokenizer 或 T5Tokenizer）。
        max_length (int): 最大序列长度。
        model_type (str): 模型类型，'gpt2' 或 't5'。

    Returns:
        dict: 分词后的输入和标签。
    """
    if 'gpt2' in model_type.lower() or 'mistral' in model_type.lower():
        # GPT-2 特有的系统提示
        # SYSTEM_PROMPT = """You are a helpful assistant. Your task is to convert casual text into formal text without changing the original meaning or altering the order of sentences. Keep the tone formal and professional, using appropriate language while ensuring that the core ideas remain intact. Please take the following casual text and rewrite it in a more formal way:
        # Casual: """
        SYSTEM_PROMPT="""Convert casual text into formal text :
        Casual: """
        inputs = [f"{SYSTEM_PROMPT}{casual} Formal:" for casual in examples['casual_text']]
        targets = examples['formal_text']

        # 分词输入
        tokenized_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )

        # 分词目标文本
        tokenized_targets = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )['input_ids']

        labels = []
        for i in range(len(examples['casual_text'])):
            # 构建输入文本和目标文本的完整 token 序列
            input_text = f"{SYSTEM_PROMPT}{examples['casual_text'][i]} Formal:"
            input_ids = tokenizer.encode(input_text, add_special_tokens=False)

            target_ids = tokenized_targets[i]

            # 计算正式文本开始的位置
            prefix_length = len(input_ids)

            # 创建标签：前缀部分为 -100，正式文本部分为实际 token IDs
            label = [-100] * prefix_length + target_ids

            # 截断或填充标签到 max_length
            if len(label) < max_length:
                label += [-100] * (max_length - len(label))
            else:
                label = label[:max_length]

            labels.append(label)

        tokenized_inputs['labels'] = labels

    elif 't5' in model_type.lower():
        # T5 特有的系统提示
        SYSTEM_PROMPT = """Convert casual text to formal text: """
        inputs = [f"{SYSTEM_PROMPT}{casual}" for casual in examples['casual_text']]
        targets = examples['formal_text']

        # 分词输入和目标文本
        tokenized_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )

        tokenized_targets = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )['input_ids']

        # T5 的标签直接是目标 token IDs
        tokenized_inputs['labels'] = tokenized_targets

    else:
        raise ValueError("Unsupported model type. Please choose 'gpt2' or 't5'.")

    return tokenized_inputs

def prepare_dataset(dataset: DatasetDict, tokenizer, max_length: int = 512, model_type: str = 'gpt2') -> DatasetDict:
    """
    应用预处理函数到整个数据集。

    Args:
        dataset (DatasetDict): 原始数据集。
        tokenizer: 分词器。
        max_length (int): 最大序列长度。
        model_type (str): 模型类型，'gpt2' 或 't5'。

    Returns:
        DatasetDict: 预处理后的数据集。
    """
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length=max_length, model_type=model_type),
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset
