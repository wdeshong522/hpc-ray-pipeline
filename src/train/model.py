"""Tokenizer, 4-bit base model and LoRA wrapper for Llama 3.1 8B Instruct.

Proposal link: Stage 1 fine-tunes Llama 3.1 8B with LoRA.  The base is held in
4-bit (DECISIONS D20) because 8B fp16 weights do not fit a 16 GB V100.
"""
import torch
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PAD_TOKEN = "<|finetune_right_pad_id|>"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def load_tokenizer(model_name):
    """Tokenizer with a pad token distinct from the end-of-turn token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise ValueError(f"{PAD_TOKEN} is not in this tokenizer's vocabulary")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("pad token must differ from the eos token")
    return tokenizer


def load_base_model(model_name, device_index):
    """Frozen 4-bit NF4 base on one GPU; fp16 compute (V100 has no bf16)."""
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        torch_dtype=torch.float16,
        device_map={"": device_index},
    )
    model.config.use_cache = False
    return model


def add_lora(model, r, alpha, dropout):
    """Prepare the 4-bit model for training and attach LoRA adapters."""
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)
