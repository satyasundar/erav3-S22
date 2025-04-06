# Gemma-3 GRPO Training

This repository contains code for fine-tuning Google's Gemma-3 1B model using GRPO (Group Relative Policy Optimization) on the GSM8K dataset. The project focuses on improving the model's mathematical reasoning capabilities.

## Overview

The project implements GRPO training on Gemma-3 1B model with the following key features:
- Fine-tuning on GSM8K dataset for mathematical reasoning
- Custom reward functions for format matching and answer correctness
- Structured output format with working out and solution sections
- Optimized training configuration using unsloth

## Model Outputs before and after GRPO

### Before GRPO
```
Q:
What is the sqrt of 101?

A:
We can estimate the square root of 101. Since $10^2 = 100$, we know that $\sqrt{101}$ is slightly greater than 10.

We can try $10.05^2 = 101.0025$, so $\sqrt{101} \approx 10.05$.

Another approach is to use the fact that $\sqrt{101} = \sqrt{101}$.
We can approximate $\sqrt{101}$ using a calculator.
$\sqrt{101} \approx 10.0498756211$.

The square root of 101 is approximately 10.05.
However, if we are looking for an exact answer, we can say that $\sqrt{101}$ is not an integer.
It is known that $\sqrt{101} \approx 10.05$.

Final Answer: The final answer is $\boxed{10.05}$
```

### After GRPO
```
Q:
What is the sqrt of 101?

A:
The square root of 101 is approximately 10.05.

Therefore, $\sqrt{101} \approx 10.05$.

Final Answer: The final answer is $\boxed{10.05}$
```

## Requirements

```bash
pip install -q huggingface_hub
pip install unsloth vllm
pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```

For Colab users, additional dependencies:
```bash
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft "trl==0.15.2" triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
```

## Model Setup

The project uses the following configuration:
- Base model: unsloth/gemma-3-1b-it
- Max sequence length: 1024
- PEFT configuration with LoRA
- GRPO training with multiple reward functions

## Training Configuration

Key training parameters:
- Learning rate: 5e-6
- Optimizer: AdamW (fused)
- Batch size: 1
- Gradient accumulation steps: 1
- Number of generations: 4
- Max prompt length: 256
- Training steps: 50

## Reward Functions

The training uses multiple reward functions:
1. `match_format_exactly`: Checks for exact format matching
2. `match_format_approximately`: Evaluates approximate format matching
3. `check_answer`: Validates answer correctness
4. `check_numbers`: Verifies numerical accuracy

## Output Format

The model is trained to produce responses in the following format:
```
<start_working_out>
[Reasoning steps]
<end_working_out>
<SOLUTION>
[Final answer]
</SOLUTION>
```

## Usage

1. Login to Hugging Face:
```python
from huggingface_hub import notebook_login
notebook_login()
```

2. Load and prepare the model:
```python
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-1b-it",
    max_seq_length = 1024,
    load_in_4bit = False,
    load_in_8bit = False,
    full_finetuning = False,
)
```

3. Configure PEFT:
```python
model = FastModel.get_peft_model(
    model,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
)
```

4. Train the model:
```python
from trl import GRPOConfig, GRPOTrainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
```

## Inference

To use the trained model for inference:
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Your question here"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 64,
    temperature = 1.0,
    top_p = 0.95,
    top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)
```

## Model Weights

The trained model weights are available at: [satyanayak/gemma-3-GRPO](https://huggingface.co/satyanayak/gemma-3-GRPO)

## License

This project is licensed under the same terms as the Gemma-3 model.
