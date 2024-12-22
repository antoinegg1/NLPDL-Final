# model.py

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch

def load_model_and_tokenizer(model_name: str, local_rank: int):
    """
    加载指定的模型和分词器，并将模型加载到指定的 GPU 上。

    Args:
        model_name (str): 模型名称或路径，例如 'gpt2' 或 't5-small'。
        local_rank (int): 当前进程的 GPU 编号。

    Returns:
        model: 加载的模型。
        tokenizer: 加载的分词器。
        model_type: 模型类型，'gpt2' 或 't5'.
    """
    if 'gpt2' in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model_type = 'gpt2'
    elif 't5' in model_name.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model_type = 't5'
    elif 'mistral' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model=AutoModelForCausalLM.from_pretrained(model_name)
        model_type = 'mistral'
    else:
        raise ValueError("Unsupported model type. Please choose 'gpt2' or 't5'.")

    # 添加填充标记（如果尚未添加）
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added pad token '[PAD]' to tokenizer for {model_type}.")


        device =torch.device('cuda', torch.cuda.current_device())
        model.to(device)
        print(f"Model loaded to device: {device}")

    return model, tokenizer, model_type
