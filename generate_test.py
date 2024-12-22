import argparse
from vllm import LLM, SamplingParams
from transformers import GPT2Tokenizer
from datasets import load_from_disk,Array3D
import tqdm
import numpy as np
import logging
import json
logging.basicConfig(level=logging.WARNING)  # 只显示 WARNING 及以上级别的日志
# SYSTEM_PROMPT = """You are a helpful assistant. Your task is to convert casual text into formal text without changing the original meaning or altering the order of sentences. Keep the tone formal and professional, using appropriate language while ensuring that the core ideas remain intact. Please take the following casual text and rewrite it in a more formal way:
# Casual: """
SYSTEM_PROMPT ="""Convert casual text to formal text: """

# import pdb; pdb.set_trace()
def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Run formal test")
    parser.add_argument("--data_set", type=str, default="/mnt/file2/changye/dataset/casual_formal_pair_ACL40k/test", help="Path to the local dataset to load using load_from_disk")
    parser.add_argument("--save_path", type=str,default="/mnt/file2/changye/NLPFINAL/result/gpt2_formal_text_result.json", help="Path to save the inference results")
    args = parser.parse_args()

    dataset_path = args.data_set
    save_path = args.save_path

    # 模型路径（请根据实际情况修改）
    GPT2_model_path = "/mnt/file2/changye/model/gpt2-formal-finetuned"  # 本地 LLaVA 模型权重路径
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_model_path)
    llm = LLM(model=GPT2_model_path, tensor_parallel_size=4)  # 根据硬件条件调整 tensor_parallel_size
    # 加载本地数据集
    local_dataset = load_from_disk(dataset_path)





    # 定义批处理输入的函数
    def prepare_inputs_batch(prompts,tokenizer):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        return inputs

    # 设置批大小，可根据显存大小进行调整
    batch_size = 8
    results = []


    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=512
    )

    # 分批处理数据集
    for i in tqdm.trange(0, len(local_dataset), batch_size):
        batch_data = local_dataset[i:i + batch_size]
        prompts =[f"{SYSTEM_PROMPT}{casual} Formal:" for casual in batch_data['casual_text']]
        # 准备批次请求
        requests = prepare_inputs_batch(prompts,tokenizer)

        # 使用 vLLM 进行推理
        outputs = llm.generate(requests, sampling_params=sampling_params)
        # breakpoint()
        # 解码输出
        decoded_outputs = [output.outputs[0].text.strip() for output in outputs]
        
        for j, output in enumerate(decoded_outputs):
            results.append({
                "casual_text": batch_data['casual_text'][j],
                "formal_text": batch_data['formal_text'][j],
                "GPT2_formal_text": output,
                "directory": batch_data['directory'][j],
                "filename": batch_data['filename'][j]
            })
            
        

    # 将结果保存到json文件中,如果没有就创建一个
    with open(save_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {save_path}")
    
        


if __name__ == "__main__":
    main()



