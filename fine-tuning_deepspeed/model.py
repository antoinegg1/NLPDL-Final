# model.py

import torch
import torch.distributed as dist
import deepspeed
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

def load_model_and_tokenizer(model_name: str, local_rank: int):
    """
    使用 DeepSpeed Zero.Init() 来并行/分片加载大模型，避免所有进程在 GPU0 堆积显存。
    """

    if not dist.is_initialized():
        deepspeed.init_distributed()

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    # ============================
    #  使用 ZeRO.Init 分片加载
    # ============================
    with deepspeed.zero.Init(remote_device="cpu"):
        tokenizer = _load_tokenizer(model_name)
        model, model_type = _load_model(model_name)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            print(f"[Rank {dist.get_rank()}] Added pad token '[PAD]' to tokenizer for {model_type}.")

    return model, tokenizer, model_type

def _load_tokenizer(model_name: str):
    if 'gpt2' in model_name.lower():
        return GPT2Tokenizer.from_pretrained(model_name, trust_remote_code=True)
    elif 't5' in model_name.lower():
        return T5Tokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def _load_model(model_name: str):
    """
    根据模型类型分别使用对应的 from_pretrained。
    """
    if 'gpt2' in model_name.lower():
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model_type = 'gpt2'
    elif 't5' in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model_type = 't5'
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model_type = model_name

    return model, model_type
