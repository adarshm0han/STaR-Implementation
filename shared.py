#!/usr/bin/env python
import os
import re
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# Base model (unchanged)
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# -------------------- HF loader --------------------
def load_base_model_full(model_name: Optional[str] = None):
    """
    Load base model with bfloat16 on GPU (if available) or float32 on CPU.
    """
    name = model_name or MODEL_NAME
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,          # <= fix deprecation: use dtype (not torch_dtype)
        device_map="auto"
    )
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok, model

# -------------------- vLLM helpers --------------------
# vLLM is optional; import lazily
def build_vllm_engine(model_name: Optional[str] = None,
                      tensor_parallel_size: int = 1,
                      gpu_memory_utilization: float = 0.9,
                      dtype: Optional[str] = None,
                      trust_remote_code: bool = True):
    """
    Create a vLLM engine. Requires `pip install vllm`.
    """
    from vllm import LLM
    name = model_name or MODEL_NAME
    engine = LLM(
        model=name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype or ("bfloat16" if torch.cuda.is_available() else "float32"),
        trust_remote_code=trust_remote_code,
        enforce_eager=True
    )
    return engine

def vllm_generate(engine, prompts: List[str], max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0):
    """
    Batched deterministic decode using vLLM. Returns list[str] of generations.
    """
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    # vLLM preserves prompt ordering
    outputs = engine.generate(prompts, sampling_params=sp)
    # Each item has .outputs[0].text
    result = []
    for out in outputs:
        if out.outputs:
            result.append(out.outputs[0].text)
        else:
            result.append("")
    return result

# -------------------- Prompt templates (unchanged) --------------------
def prompt_cot(question: str) -> str:
    sys = "You are a careful math tutor. Solve step by step. End with: #### <number>"
    return f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\nQ: {question}\nA: [/INST]"

def prompt_with_hint(question: str, gold: str) -> str:
    sys = (
        "You already know the correct final answer, but DO NOT mention that fact."
        " Produce a clear step-by-step rationale that independently leads to it."
        f" End exactly with: #### {gold}."
    )
    return f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\nQ: {question}\nA: [/INST]"

# -------------------- Parsing helpers (unchanged) --------------------
def gsm8k_extract_gold(answer_text: str):
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text.strip())
    return m.group(1) if m else None

def extract_pred(model_output: str):
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", model_output)
    return m[-1] if m else None

def decode_new_text(tok, prompt: str, generated_ids):
    text = tok.decode(generated_ids[0], skip_special_tokens=True)
    return text[len(prompt):] if text.startswith(prompt) else text
