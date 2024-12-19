# Proposal for NLP Final Project: Text Style Transfer with LLMs

2200017853 李长烨

## Motivation

Academic writing can be challenging for students, often resulting in drafts that are rough or inconsistent in style. Automating the refinement of such writing through Text Style Transfer (TST) offers a practical solution to enhance clarity, coherence, and adherence to academic conventions.

## Related Work

Text Style Transfer (TST) has gained significant attention in recent years, particularly with the rise of large language models (LLMs). Relevant works include:

1. **Text Style Transfer Overview**: An introduction to the fundamental techniques and challenges in the field.  
   [[2407.14822] Text Style Transfer: An Introductory Overview](https://arxiv.org/abs/2407.14822)

2. **Transformers-based TST Approaches**: Recent advancements leveraging transformers-based models for style adaptation across diverse text corpora.

## Methodology

### Dataset
We will utilize publicly available datasets suitable for TST tasks, including:
- **[WINGNUS/ACL-OCL](https://huggingface.co/datasets/WINGNUS/ACL-OCL)**: A dataset for pretraining in academic contexts.  
- **[passeul/style-transfer-model-evaluation](https://github.com/passeul/style-transfer-model-evaluation)**: A collection for evaluating style transfer models.  
- **NiGuLa/SGDD-TST**: A dataset designed for fine-tuning and testing TST models.  

### Approach
1. **Fine-tuning**: Leverage pre-trained models such as T5-large and GPT-2-large, adapting them to the specific task of academic text style refinement.  
2. **Prompt Engineering**: Design prompts tailored to guide LLMs in generating outputs that align with academic writing conventions.  
3. **Pre-training**: Experiment with additional pre-training on domain-specific corpora to enhance the models' understanding of academic styles.

### Models
- **T5-large**: A transformer-based model designed for sequence-to-sequence tasks.  
- **GPT-2-large**: A generative pre-trained transformer capable of producing coherent and stylistically consistent text.

### Evaluation
- **NiGuLa/SGDD-TST**: Evaluate model performance using this benchmark dataset.  
- **GPT-4 Feedback**: Leverage GPT-4 as a secondary evaluator to provide qualitative insights into output quality.  
- **AI-Based Comparison**: Use automated metrics and human-assisted tools to compare model outputs.

## Contribution

This project aims to achieve the following:
1. **Development of TST Models**: Fine-tune and adapt existing LLMs for academic text style refinement.  
2. **Comparative Evaluation**: Analyze and compare the performance of T5-based and GPT-based architectures on the TST task.  
3. **Impact**: Provide insights into the strengths and limitations of LLMs in handling nuanced style transfer tasks, particularly in academic contexts.  

By advancing TST methodologies and offering a structured evaluation, this project has the potential to contribute valuable tools and techniques for automated academic writing enhancement.
