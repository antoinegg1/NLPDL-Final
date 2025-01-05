from datasets import load_from_disk, concatenate_datasets
import hashlib
import re
import numpy as np
from datasets import Dataset

def load_and_combine_datasets(paths):
    """
    Load multiple datasets from disk and combine them into one.

    Args:
        paths (list): List of paths to the datasets.

    Returns:
        Dataset: Combined dataset.
    """
    datasets = [load_from_disk(path) for path in paths]
    return concatenate_datasets(datasets)

def split_dataset(dataset, test_size=0.2, val_split=0.5, seed=42):
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset (Dataset): Combined dataset.
        test_size (float): Proportion of the test dataset.
        val_split (float): Proportion of the validation set from the test set.
        seed (int): Random seed.

    Returns:
        dict: Dictionary containing train, validation, and test datasets.
    """
    train_test_split = dataset.train_test_split(test_size=test_size, seed=seed)
    val_test_split = train_test_split['test'].train_test_split(test_size=val_split, seed=seed)

    return {
        'train': train_test_split['train'],
        'val': val_test_split['train'],
        'test': val_test_split['test']
    }

def save_datasets(datasets, output_dir):
    """
    Save train, validation, and test datasets to disk.

    Args:
        datasets (dict): Dictionary containing train, validation, and test datasets.
        output_dir (str): Directory to save the datasets.
    """
    datasets['train'].save_to_disk(f"{output_dir}/train")
    datasets['val'].save_to_disk(f"{output_dir}/val")
    datasets['test'].save_to_disk(f"{output_dir}/test")

def generate_text_hash(text):
    """
    Generate a unique identifier for the given text using SHA-256.

    Args:
        text (str): Input text.

    Returns:
        str: Unique hash for the text.
    """
    hash_object = hashlib.sha256(text.encode('utf-8'))
    return hash_object.hexdigest()

def clean_text_english(text):
    """
    Clean and normalize English text.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'（.*?）', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'https?\s*:\s*//(?:\S+\s*)*', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def is_valid_sentence(sentence):
    """
    Check if a sentence is valid.

    Args:
        sentence (str): Sentence to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not sentence:
        return False
    if re.match(r'^https?\s*:\s*//', sentence) or re.match(r'^\d+$', sentence):
        return False
    if not re.search(r'[a-zA-Z]', sentence):
        return False
    return True

def split_and_process_texts(text_list, max_length=200):
    """
    Process and split texts into smaller chunks.

    Args:
        text_list (list): List of texts.
        max_length (int): Maximum length of each text chunk.

    Returns:
        list: Processed and split texts.
    """
    updated_text_list = []

    for text in text_list:
        text['text'] = clean_text_english(text['text'])
        
        if len(text['text']) > max_length:
            sentences = re.split(r'(?<=[.!?])\s+', text['text'])
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if is_valid_sentence(sentence):
                    if len(current_chunk) + len(sentence) > max_length:
                        updated_text_list.append({
                            'directory': text['directory'],
                            'filename': text['filename'],
                            'text': current_chunk.strip(),
                            'hash': generate_text_hash(text['directory'] + text['filename'] + current_chunk.strip())
                        })
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence

            if current_chunk:
                updated_text_list.append({
                    'directory': text['directory'],
                    'filename': text['filename'],
                    'text': current_chunk.strip(),
                    'hash': generate_text_hash(text['directory'] + text['filename'] + current_chunk.strip())
                })
        else:
            updated_text_list.append({
                'directory': text['directory'],
                'filename': text['filename'],
                'text': text['text'].strip(),
                'hash': generate_text_hash(text['directory'] + text['filename'] + text['text'].strip())
            })

    return updated_text_list

def calculate_evaluation_metrics(dataset_path, pattern):
    """
    Calculate evaluation metrics from dataset.

    Args:
        dataset_path (str): Path to the dataset.
        pattern (str): Pattern to extract evaluation metrics.

    Returns:
        tuple: Averages of Style Transfer Strength, Content Preservation, and Fluency.
    """
    dataset = load_from_disk(dataset_path)

    style_transfer_strength = []
    content_preservation = []
    fluency = []

    for item in dataset:
        matches = re.findall(pattern, item['evaluation'])
        if len(matches) == 3:
            style_transfer_strength.append(float(matches[0]))
            content_preservation.append(float(matches[1]))
            fluency.append(float(matches[2]))

    return (
        np.array(style_transfer_strength).mean(),
        np.array(content_preservation).mean(),
        np.array(fluency).mean()
    )
