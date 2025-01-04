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
    移除 low_cpu_mem_usage=True, 以免与 ZeRO Init 的分片逻辑冲突导致 "meta tensor" 报错。

    在 ds_config.json 中启用 ZeRO Stage 2/3，会大大减少单卡显存占用。
    """

    # 如果还未初始化分布式环境，就先进行初始化
    if not dist.is_initialized():
        deepspeed.init_distributed()

    # 设置当前进程使用的 GPU 设备 (单机多卡)
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    # ============================
    #  使用 ZeRO.Init 分片加载
    # ============================
    # remote_device="cpu" 表示在加载权重时先放到 CPU，
    # 再由 DeepSpeed 分片到各 GPU，避免一开始就挤占GPU0。
    # 要结合 ds_config.json 的 ZeRO stage，效果更好。
    with deepspeed.zero.Init(remote_device="cpu"):
        # 先加载分词器(占内存小)，再加载大模型(占内存大)。
        # 去掉 low_cpu_mem_usage=True，以避免与 meta device 相关冲突。
        tokenizer = _load_tokenizer(model_name)
        model, model_type = _load_model(model_name)

        # 如果 tokenizer 缺少 pad_token，则添加
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            print(f"[Rank {dist.get_rank()}] Added pad token '[PAD]' to tokenizer for {model_type}.")

    # 此时，模型的参数已由 DeepSpeed Zero.Init() 按需分配，避免堆在 GPU0。
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
    移除 low_cpu_mem_usage=True，避免 meta device 引起冲突。
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
