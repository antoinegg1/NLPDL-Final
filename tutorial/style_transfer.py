
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel,AutoTokenizer,AutoModelForCausalLM
import torch
from tqdm import tqdm

model_type = ""

if model_type == "gpt2":
    model_path="/mnt/file2/changye/model/gpt2-formal-finetuned"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
elif model_type == "t5":
    model_path="/mnt/file2/changye/model/fine_tuned_t5/fine_tuned_t5"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
        # 定义模型和优化器
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    # checkpoint = torch.load("/mnt/file2/changye/model/fine_tuned_t5/checkpoint_500.pt")  # 加载文件
    # model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型
else:
    n_devices = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path="/mnt/file2/changye/model/NLP/qwen-formal-trained"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model=AutoModelForCausalLM.from_pretrained(model_path,device_map='auto')
    model_type=model_path
    # model.to(device)

def generate_text(prompt, max_length=512):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    attention_mask = torch.ones(input_ids.shape).to(input_ids.device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)
SYSTEM_PROMPT = """You are a helpful assistant. Your task is to convert casual text into formal text without changing the original meaning or altering the order of sentences. Keep the tone formal and professional, using appropriate language while ensuring that the core ideas remain intact. Please take the following casual text and rewrite it in a more formal way:
Casual: """
casual = "We present a novel algorithm for Japanese dependency analysis. The algorithm allows us to analyze dependency structures of a sentence in linear-time while keeping a state-of-the-art accuracy. In this paper, we show a formal description of the algorithm and discuss it theoretically with respect to time complexity. In addition, we evaluate its efficiency and performance empirically against the Kyoto University Corpus. The proposed algorithm with improved models for dependency yields the best accuracy in the previously published results on the Kyoto University Corpus." # input text
inputs = f"{SYSTEM_PROMPT}{casual}  Formal:"
output_text = generate_text(inputs)
print(output_text)