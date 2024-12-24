import argparse
import math
import torch
import nltk
from collections import Counter
from rouge import Rouge
from bert_score import score
from datasets import load_from_disk
# Transformers imports
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
import tqdm
# Download NLTK data for tokenization and BLEU if not already available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')
############################
# 1. Define Evaluation Functions
############################

def calculate_bleu(generated, reference):
    """
    Calculates the average sentence-level BLEU score between generated and reference texts.
    """
    generated_tokens = [nltk.word_tokenize(sent) for sent in generated]
    reference_tokens = [[nltk.word_tokenize(sent)] for sent in reference]
    
    bleu_scores = []
    for ref, gen in zip(reference_tokens, generated_tokens):
        bleu_scores.append(nltk.translate.bleu_score.sentence_bleu(ref, gen))
    
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

def calculate_rouge(generated, reference):
    """
    Calculates average ROUGE scores (1, 2, and L) between generated and reference texts.
    """
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference, avg=True)
    return scores

def calculate_perplexity(model, tokenizer, texts, is_seq2seq=False):
    """
    Calculates perplexity for given texts using a language model.
    For Seq2Seq models like T5, we typically use the encoder-decoder approach.
    For GPT-2 or other causal models, we use standard LM approach.
    """
    # Tokenize
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodings.input_ids.to(model.device)

    with torch.no_grad():
        if is_seq2seq:
            # For T5 or other Seq2Seq: 
            # We can pass input_ids as both encoder and decoder inputs 
            # for a quick approximation of perplexity.
            outputs = model(input_ids=input_ids, labels=input_ids)
        else:
            # Causal LM (e.g., GPT-2)
            outputs = model(input_ids, labels=input_ids)
        
        loss = outputs.loss
    return math.exp(loss.item()) if loss.item() < 100 else float('inf')

def calculate_bert_score(generated, reference, lang="en"):
    """
    Calculates BERTScore (Precision, Recall, F1) for generated vs. reference texts.
    """
    P, R, F1 = score(generated, reference, lang=lang, verbose=False)
    return {
        "Precision": P.mean().item(), 
        "Recall": R.mean().item(), 
        "F1": F1.mean().item()
    }

def calculate_diversity(generated):
    """
    Calculates Distinct-n and repetition rate for n-grams.
    """
    total_tokens = sum(len(sent.split()) for sent in generated)
    if total_tokens == 0:
        return {"Distinct-n": {}, "Repetition Rate": {}}
    
    all_ngrams = {n: Counter() for n in range(1, 5)}
    unique_ngrams = {n: set() for n in range(1, 5)}

    for sent in generated:
        tokens = sent.split()
        for n in range(1, 5):
            ngrams = list(nltk.ngrams(tokens, n))
            all_ngrams[n].update(ngrams)
            unique_ngrams[n].update(ngrams)
    
    distinct_n = {}
    repetition_rate = {}
    
    for n in range(1, 5):
        if total_tokens > 0:
            distinct_n[n] = len(unique_ngrams[n]) / total_tokens
        else:
            distinct_n[n] = 0.0
        
        if len(all_ngrams[n]) > 0:
            repetition_rate[n] = 1 - (len(unique_ngrams[n]) / len(all_ngrams[n]))
        else:
            repetition_rate[n] = 0.0
    
    return {
        "Distinct-n": distinct_n, 
        "Repetition Rate": repetition_rate
    }

def calculate_length_normalization(generated, reference):
    """
    Computes the ratio of the generated text length to the reference text length.
    """
    if not reference:
        return {"Average Length Ratio": 0.0}
    
    lengths_gen = [len(sent.split()) for sent in generated]
    lengths_ref = [len(sent.split()) for sent in reference]
    
    length_ratios = []
    for gen_len, ref_len in zip(lengths_gen, lengths_ref):
        if ref_len > 0:
            length_ratios.append(gen_len / ref_len)
    
    if len(length_ratios) == 0:
        return {"Average Length Ratio": 0.0}
    else:
        return {"Average Length Ratio": sum(length_ratios) / len(length_ratios)}

############################
# 2. Load Models Dynamically
############################

def load_model(model_type, model_name):
    """
    Dynamically loads tokenizer and model based on model type:
    - t5: T5Tokenizer + T5ForConditionalGeneration
    - gpt2: GPT2Tokenizer + GPT2LMHeadModel
    - auto: Use AutoTokenizer + AutoModelForCausalLM (or AutoModelForSeq2SeqLM if it's a seq2seq model)
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        return tokenizer, model, True  # (tokenizer, model, is_seq2seq = True)
    
    elif model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        # GPT-2 does not have a pad token by default, so we set EOS_token as pad if needed
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
        return tokenizer, model, False
    
    else:  # "auto"
        # Decide if we want a causal or seq2seq model 
        # For demonstration, we'll just assume it's a causal model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # If your model_name is a T5 or BART, you might use AutoModelForSeq2SeqLM
        # Otherwise, default to AutoModelForCausalLM. Adjust logic as needed.
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(device)
            return tokenizer, model, True
        except:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(device)
            return tokenizer, model, False

############################
# 3. Main Function
############################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["t5", "gpt2", "auto"], default="gpt2",
                        help="Type of model to load: t5, gpt2, or auto.")
    parser.add_argument("--model_name", type=str, default="/mnt/file2/changye/model/gpt2-formal-finetuned_short_prompt/checkpoint-4000",
                        help="Name or path of the model, e.g., 't5-small', 'gpt2', 'facebook/bart-base', etc.")
    parser.add_argument("--dataset_path", type=str,
                        default="/mnt/file2/changye/dataset/casual_formal_sentence_pair_ACL170k/test",
                        help="Path to the test dataset.")
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Number of test examples to evaluate (for demonstration).")
    args = parser.parse_args()

    # 3.1 Load model & tokenizer
    tokenizer, model, is_seq2seq = load_model(args.model_type, args.model_name)
    model.eval()

    # 3.2 Load dataset (simplified)
    #    Here, we assume each line in the dataset is a pair: "casual_text\tformal_text"
    #    Adjust parsing logic to match your actual data format.
    dataset=load_from_disk(args.dataset_path)
    # Safety check


    formal_texts = [example['formal_text'] for example in dataset]
    sample_size = min(args.num_examples, len(formal_texts))
    formal_texts = formal_texts[:sample_size]
    ############################
    # 3.3 Generate Outputs
    ############################
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generated_texts = []
    for text in tqdm.tqdm(formal_texts, desc="Generating Texts"):
        min_len=min(len(text),50)
    
        input_text=text[:min_len]
        if is_seq2seq:
            # For T5 or Seq2Seq models, we typically prefix tasks.
            # Example for T5: "translate English to Formal: <text>"
            # Adjust according to your specific fine-tuning approach
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            attention_mask = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
            output_ids = model.generate(input_ids, max_length=64)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            # For GPT-2 or causal models
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            attention_mask = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
            output_ids = model.generate(input_ids, max_length=64, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        generated_texts.append(output_text)

    ############################
    # 3.4 Evaluate
    ############################
    # BLEU & ROUGE & BERTScore typically compare generated_texts with formal_texts
    # (assuming the formal_text is the "reference" for the casual_text).
    
    bleu_score = calculate_bleu(generated_texts, formal_texts)
    rouge_scores = calculate_rouge(generated_texts, formal_texts)
    ppl = calculate_perplexity(model, tokenizer, generated_texts, is_seq2seq=is_seq2seq)
    bert_scores = calculate_bert_score(generated_texts, formal_texts)
    diversity = calculate_diversity(generated_texts)
    length_norm = calculate_length_normalization(generated_texts, formal_texts)

    ############################
    # 3.5 Print Results
    ############################
    print("========== Evaluation Results ==========")
    print(f"Model Type: {args.model_type}")
    print(f"Model Name: {args.model_name}")
    print(f"Data Path: {args.dataset_path}")
    print("----------------------------------------")
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"ROUGE Scores: {rouge_scores}")
    print(f"Perplexity: {ppl:.4f}")
    print(f"BERTScore: {bert_scores}")
    print(f"Diversity: {diversity}")
    print(f"Length Normalization: {length_norm}")
    print("========================================")

    # # Optionally print some sample generations for inspection
    # print("\nSample Generations:")
    # for i in range(num_samples):
    #     print(f"\nCasual Text: {casual_texts[i]}")
    #     print(f"Generated Text: {generated_texts[i]}")
    #     print(f"Reference (Formal): {formal_texts[i]}")

if __name__ == "__main__":
    main()
