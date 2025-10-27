# star_bootstrap_iterative.py
# Fully self-contained iterative STaR bootstrapping script with vLLM + SFT

import os, sys, re, json, time, math, gc
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Import shared prompting utils
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from shared import prompt_cot, prompt_with_hint, gsm8k_extract_gold, build_vllm_engine, vllm_generate

# ------------------ Parameters ------------------
ROUNDS = 5
HINT_PASSES = 4
BATCH = 256
MAX_NEW = 512
OUT_PATH = "Data/star_train_from_iterative.jsonl"
SFT_OUTPUT_DIR = "Models/star_sft"
INITIAL_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# ------------------ Helpers ------------------
def clean_rationale(text):
    text = text.replace("<<SYS>>", "").replace("<</SYS>>", "")
    text = re.sub(r"\[/?INST\]", "", text)
    m = re.search(r"####\s*-?\d[\d,]*\.?\d*", text)
    if m:
        text = text[:m.start()]
    return text.strip()

def ensure_final_answer(text, answer):
    return text.strip() + f"\n#### {answer}"

def matches_answer(text, gold):
    if not gold:
        return False
    matches = re.findall(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    return matches and matches[-1].replace(",", "") == gold


# ------------------ SFT Helpers ------------------
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = f"<s>[INST] <<SYS>>\nYou are a careful math tutor. Solve step by step. End with: #### <number>\n<</SYS>>\nQ: {r['question']}\nA: [/INST]"
        target = ensure_final_answer(clean_rationale(r["rationale"]), r["answer"])
        full_text = prompt + target
        enc = self.tok(full_text, truncation=True, max_length=self.max_len, padding=False, add_special_tokens=False)
        n_prompt = len(self.tok(prompt, add_special_tokens=False)["input_ids"])
        input_ids = enc["input_ids"]
        labels = [-100] * n_prompt + input_ids[n_prompt:]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

# ------------------ STaR Logic ------------------
def run_star_loop(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("gsm8k", "main", split="train")
    qs = [x["question"].strip() for x in dataset]
    golds = [gsm8k_extract_gold(x["answer"]) for x in dataset]
    cumulative = []

    for round_id in range(1, ROUNDS+1):
        print(f"\n=== STaR Round {round_id} ===")
        seen = set(r["question"] for r in cumulative)
        remain_idxs = [i for i, q in enumerate(qs) if q not in seen]
        if not remain_idxs:
            print("[done] All examples bootstrapped.")
            break

        # --- Generate rationales using vLLM ---
        engine = build_vllm_engine(model_name)
        prompts = [prompt_cot(qs[i]) for i in remain_idxs]
        decodes = vllm_generate(engine, prompts, max_new_tokens=MAX_NEW)

        accepted = []
        for i, out in zip(remain_idxs, decodes):
            if matches_answer(out, golds[i]):
                accepted.append({"question": qs[i], "rationale": clean_rationale(out), "answer": golds[i]})

        # --- Add smart hints for failed questions ---
        remain_idxs = [i for i in remain_idxs if qs[i] not in set(r["question"] for r in accepted)]
        for _ in range(HINT_PASSES):
            if not remain_idxs: break
            hint_prompts = [prompt_with_hint(qs[i], golds[i]) for i in remain_idxs]
            hint_decodes = vllm_generate(engine, hint_prompts, max_new_tokens=MAX_NEW)
            for i, out in zip(remain_idxs, hint_decodes):
                if matches_answer(out, golds[i]):
                    accepted.append({"question": qs[i], "rationale": clean_rationale(out), "answer": golds[i]})
            remain_idxs = [i for i in remain_idxs if qs[i] not in set(r["question"] for r in accepted)]

        print(f"Round {round_id} accepted {len(accepted)} new examples.")
        cumulative += accepted

        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for r in cumulative:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # --- Release vLLM + clear GPU ---
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1.0)

        # --- Fine-tune on current examples (SFT) ---
        print("[train] Fine-tuning model on cumulative rationales...")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_set = SFTDataset(cumulative, tokenizer, max_len=1024)
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")
        args = TrainingArguments(
            output_dir=SFT_OUTPUT_DIR,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=10,
            num_train_epochs=1,
            learning_rate=2e-5,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_set,
            tokenizer=tokenizer,
            data_collator=collator
        )
        trainer.train()
        trainer.save_model(SFT_OUTPUT_DIR)
        tokenizer.save_pretrained(SFT_OUTPUT_DIR)

        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1.0)

        model_name = SFT_OUTPUT_DIR  # use fine-tuned model for next round

    print("\u2705 Done. Final dataset and model written.")

if __name__ == "__main__":
    run_star_loop(INITIAL_MODEL)
