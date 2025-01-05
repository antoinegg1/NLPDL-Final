# Proposal for NLP Final Project: Text Style Transfer with LLMs

2200017853 李长烨

## Abstract

我们从ACL文章构建了一个预训练数据集和一个偏好数据集，用于训练从非正式文本到ACL风格的正式文本的迁移，我们在T5模型和GPT2模型上进行了训练，并且我们构建了评测集，通过AI feedback评测风格迁移的质量，我们对训练好的T5模型，GPT2模型和通用的一个大模型mistral7b模型进行了评测

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


## evaluation result

### automate
========== Evaluation Results ==========
Model Type: gpt2
Model Name: /mnt/file2/changye/model/gpt2-formal-finetuned_short_prompt/checkpoint-4000
Data Path: /mnt/file2/changye/dataset/casual_formal_pair_ACL40k/test
----------------------------------------
BLEU Score: 0.1066
ROUGE Scores: {'rouge-1': {'r': 0.2901783762995762, 'p': 0.39728214144495133, 'f': 0.29979913104882366}, 'rouge-2': {'r': 0.1864647756365796, 'p': 0.22873244688008956, 'f': 0.17361335854152757}, 'rouge-l': {'r': 0.28352213243550917, 'p': 0.38727167196709034, 'f': 0.29230421642973614}}
Perplexity: 9.0371
BERTScore: {'Precision': 0.8551714420318604, 'Recall': 0.8433540463447571, 'F1': 0.848703145980835}
Diversity: {'Distinct-n': {1: 0.2517834662190516, 2: 0.5493075954678976, 3: 0.6783466219051616, 4: 0.7310113302559799}, 'Repetition Rate': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
Length Normalization: {'Average Length Ratio': 1.0679772153113702}
========================================
Model Type: gpt2
Model Name: /mnt/file2/changye/model/gpt2-large
Data Path: /mnt/file2/changye/dataset/casual_formal_pair_ACL40k/test
BLEU Score: 0.1113
ROUGE Scores: {'rouge-1': {'r': 0.2755501660204627, 'p': 0.49452518394649203, 'f': 0.3104046359244537}, 'rouge-2': {'r': 0.18706580495879016, 'p': 0.2944180198047594, 'f': 0.19013755284954736}, 'rouge-l': {'r': 0.271100427555902, 'p': 0.48495755492695464, 'f': 0.30479218521834417}}
Perplexity: 6.1339
BERTScore: {'Precision': 0.845746636390686, 'Recall': 0.8364123702049255, 'F1': 0.8404953479766846}
Diversity: {'Distinct-n': {1: 0.26366898881148404, 2: 0.5330377876293012, 3: 0.6554781507283091, 4: 0.6896770107663078}, 'Repetition Rate': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
Length Normalization: {'Average Length Ratio': 1.0210514291375687}

========== Evaluation Results ==========
Model Type: gpt2
Model Name: /mnt/file2/changye/model/fine_tuned_gpt2
Data Path: /mnt/file2/changye/dataset/casual_formal_pair_ACL40k/test
----------------------------------------
BLEU Score: 0.0984
ROUGE Scores: {'rouge-1': {'r': 0.2879051908486429, 'p': 0.48625790276958325, 'f': 0.3171666368112837}, 'rouge-2': {'r': 0.18737824523790667, 'p': 0.29591344915432055, 'f': 0.18298151929295436}, 'rouge-l': {'r': 0.28038221169176203, 'p': 0.4717211493319388, 'f': 0.30778125727697936}}
Perplexity: 5.3937
BERTScore: {'Precision': 0.8643019199371338, 'Recall': 0.8424625992774963, 'F1': 0.8526028990745544}
Diversity: {'Distinct-n': {1: 0.2467202708421498, 2: 0.5588235294117647, 3: 0.7200592467202709, 4: 0.7771900126957257}, 'Repetition Rate': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
Length Normalization: {'Average Length Ratio': 1.0650099097572063}
========================================
========== Evaluation Results ==========
Model Type: gpt2
Model Name: /mnt/file2/changye/model/fine_tuned_gpt2
Data Path: /mnt/file2/changye/dataset/casual_formal_sentence_pair_ACL170k/test
----------------------------------------
BLEU Score: 0.1561
ROUGE Scores: {'rouge-1': {'r': 0.3969971778290537, 'p': 0.44131888341105985, 'f': 0.3940078096487681}, 'rouge-2': {'r': 0.2702746917571915, 'p': 0.2808448345814971, 'f': 0.24939973231273352}, 'rouge-l': {'r': 0.3880521989894386, 'p': 0.43102568808409075, 'f': 0.38484035567891545}}
Perplexity: 5.2115
BERTScore: {'Precision': 0.8579785823822021, 'Recall': 0.8713116645812988, 'F1': 0.8641879558563232}
Diversity: {'Distinct-n': {1: 0.2517879680269247, 2: 0.5532183424484645, 3: 0.720445940260833, 4: 0.7787126630206143}, 'Repetition Rate': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
Length Normalization: {'Average Length Ratio': 1.6549784895740496}
========================================
========== Evaluation Results ==========
Model Type: gpt2
Model Name: /mnt/file2/changye/model/gpt2-formal-finetuned_short_prompt/checkpoint-4000
Data Path: /mnt/file2/changye/dataset/casual_formal_sentence_pair_ACL170k/test
----------------------------------------
BLEU Score: 0.1577
ROUGE Scores: {'rouge-1': {'r': 0.3999664559464171, 'p': 0.34855727521881535, 'f': 0.3529024212016193}, 'rouge-2': {'r': 0.26314291541414025, 'p': 0.21207068146688335, 'f': 0.21893502407446142}, 'rouge-l': {'r': 0.387864154162954, 'p': 0.33770273072734375, 'f': 0.34187359726888394}}
Perplexity: 8.8030
BERTScore: {'Precision': 0.8553260564804077, 'Recall': 0.8733161687850952, 'F1': 0.8639866709709167}
Diversity: {'Distinct-n': {1: 0.2594752186588921, 2: 0.5689296126613911, 3: 0.7142857142857143, 4: 0.7634319033735943}, 'Repetition Rate': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
Length Normalization: {'Average Length Ratio': 1.7640002277178646}
========================================
========== Evaluation Results ==========
Model Type: gpt2
Model Name: /mnt/file2/changye/model/gpt2-large
Data Path: /mnt/file2/changye/dataset/casual_formal_sentence_pair_ACL170k/test
----------------------------------------
BLEU Score: 0.1639
ROUGE Scores: {'rouge-1': {'r': 0.3801270372039081, 'p': 0.40836387126262436, 'f': 0.37682656657361113}, 'rouge-2': {'r': 0.2621909106739208, 'p': 0.24784176838846594, 'f': 0.24124100676403226}, 'rouge-l': {'r': 0.37377557339596407, 'p': 0.40134997844636827, 'f': 0.3702566387049073}}
Perplexity: 6.1114
BERTScore: {'Precision': 0.8521018028259277, 'Recall': 0.8656085133552551, 'F1': 0.8585867285728455}
Diversity: {'Distinct-n': {1: 0.2679535076795351, 2: 0.562266500622665, 3: 0.7027812370278124, 4: 0.7424242424242424}, 'Repetition Rate': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
Length Normalization: {'Average Length Ratio': 1.6308074362596552}
========================================
========== Evaluation Results ==========
Model Type: auto
Model Name: /mnt/file2/changye/model/NLP/Qwen2.5-1.5B-Instruct-finetune/checkpoint-2000
Data Path: /mnt/file2/changye/dataset/NLP/casual_formal_sentence_pair_ACL170k/test
----------------------------------------
BLEU Score: 0.1588
ROUGE Scores: {'rouge-1': {'r': 0.4264435812070487, 'p': 0.2753976325161302, 'f': 0.32013358371939815}, 'rouge-2': {'r': 0.26101975549771805, 'p': 0.1700963846272772, 'f': 0.1952260227839123}, 'rouge-l': {'r': 0.4123957358694215, 'p': 0.2659854026358996, 'f': 0.3089675626398913}}
Perplexity: 6.7583
BERTScore: {'Precision': 0.8550587296485901, 'Recall': 0.8775673508644104, 'F1': 0.8659509420394897}
Diversity: {'Distinct-n': {1: 0.3589848546868604, 2: 0.8063855914858781, 3: 0.9267294310274253, 4: 0.9287760949652067}, 'Repetition Rate': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
Length Normalization: {'Average Length Ratio': 1.7847899381941124}
========================================
========== Evaluation Results ==========
Model Type: auto
Model Name: /mnt/file2/changye/model/NLP/Qwen2.5-1.5B-Instruct-finetune/checkpoint-2000
Data Path: /mnt/file2/changye/dataset/NLP/casual_formal_pair_ACL40k/test
----------------------------------------
BLEU Score: 0.1061
ROUGE Scores: {'rouge-1': {'r': 0.323586181617023, 'p': 0.3190045015712839, 'f': 0.2868051269922186}, 'rouge-2': {'r': 0.1912540161239054, 'p': 0.1736544556465611, 'f': 0.15265749853120514}, 'rouge-l': {'r': 0.3125053970019196, 'p': 0.30577102693616587, 'f': 0.27557582403429204}}
Perplexity: 7.4155
BERTScore: {'Precision': 0.8583482503890991, 'Recall': 0.8496290445327759, 'F1': 0.8534549474716187}
Diversity: {'Distinct-n': {1: 0.35640495867768596, 2: 0.8020661157024793, 3: 0.9185950413223141, 4: 0.9262396694214876}, 'Repetition Rate': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
Length Normalization: {'Average Length Ratio': 1.1150491580156325}
========================================
========== Evaluation Results ==========
Model Type: auto
Model Name: /mnt/file2/changye/model/NLP/Qwen2.5-1.5B-Instruct
Data Path: /mnt/file2/changye/dataset/NLP/casual_formal_pair_ACL40k/test
----------------------------------------
BLEU Score: 0.1022
ROUGE Scores: {'rouge-1': {'r': 0.3112848682316797, 'p': 0.3150193521231086, 'f': 0.27961981026371313}, 'rouge-2': {'r': 0.1843446394824244, 'p': 0.16971772660040177, 'f': 0.14867988708784302}, 'rouge-l': {'r': 0.30194884069617084, 'p': 0.3013733176135041, 'f': 0.2691745893306191}}
Perplexity: 7.4654
BERTScore: {'Precision': 0.8468213081359863, 'Recall': 0.8428120613098145, 'F1': 0.8443233966827393}
Diversity: {'Distinct-n': {1: 0.401281786231135, 2: 0.8250982013644821, 3: 0.9216456481290056, 4: 0.9226793467025015}, 'Repetition Rate': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
Length Normalization: {'Average Length Ratio': 1.1097649374273808}
========================================
========== Evaluation Results ==========
Model Type: auto
Model Name: /mnt/file2/changye/model/NLP/Qwen2.5-1.5B-Instruct
Data Path: /mnt/file2/changye/dataset/NLP/casual_formal_sentence_pair_ACL170k/test
----------------------------------------
BLEU Score: 0.1594
ROUGE Scores: {'rouge-1': {'r': 0.421630482390337, 'p': 0.2773633525958341, 'f': 0.32002718230467453}, 'rouge-2': {'r': 0.26247361678976733, 'p': 0.1735307065411744, 'f': 0.19680727311444343}, 'rouge-l': {'r': 0.4041896053443159, 'p': 0.2644900463979086, 'f': 0.30545955873912184}}
Perplexity: 7.9291
BERTScore: {'Precision': 0.8441570997238159, 'Recall': 0.8710609674453735, 'F1': 0.8571165204048157}
Diversity: {'Distinct-n': {1: 0.39652677279305354, 2: 0.8296464750878644, 3: 0.9295017572875749, 4: 0.9278478395699814}, 'Repetition Rate': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
Length Normalization: {'Average Length Ratio': 1.7652227173084476}
========================================


### AI feedback

mistral paragraph
Style Transfer Strength:  0.7234793491864832
Content Preservation:  0.8937421777221529
Fluency:  0.8588986232790992

gpt2-finetuned on paragraph
Style Transfer Strength:  0.24415519399249064
Content Preservation:  0.19118898623279104
Fluency:  0.32615769712140175

gpt2-finetuned on sentence
Style Transfer Strength:  0.22100685733400235
Content Preservation:  0.19541729386184983
Fluency:  0.33191169091821365

qwen-sentence
Style Transfer Strength:  0.5504504504504505
Content Preservation:  0.7384384384384385
Fluency:  0.6790790790790792

qwen-paragraph
Style Transfer Strength:  0.6180602006688962
Content Preservation:  0.7856187290969899
Fluency:  0.7110367892976589


mistral on sentence
Style Transfer Strength:  0.7329999999999999
Content Preservation:  0.8970000000000001
Fluency:  0.8736666666666667

gpt2 on paragraph
Style Transfer Strength:  0.2461279461279461
Content Preservation:  0.20875420875420875
Fluency:  0.3501683501683502

gpt2 on sentence
Style Transfer Strength:  0.26107382550335567
Content Preservation:  0.2389261744966443
Fluency:  0.4003355704697987

gpt2-pretrained on paragraph
Style Transfer Strength:  0.2622895622895623
Content Preservation:  0.22356902356902358
Fluency:  0.3727272727272727


gpt2-pretrained on sentence
Style Transfer Strength:  0.24040404040404043
Content Preservation:  0.21346801346801347
Fluency:  0.39124579124579123