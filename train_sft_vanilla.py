#!/usr/bin/env python
import os, sys, math, json, torch, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from shared import MODEL_NAME

INSTR = "<s>[INST] <<SYS>>\nYou are a careful math tutor. Solve step by step. End with: #### <number>\n<</SYS>>\n"

class SFTJsonl(Dataset):
    def __init__(self, path, tok, max_len=1024):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tok = tok
        self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        ex = self.rows[i]
        prompt = f"{INSTR}Q: {ex['question']}\nA: [/INST]"
        target = ex["answer"]
        enc_prompt = self.tok(prompt, add_special_tokens=False)
        enc_full   = self.tok(prompt + target, add_special_tokens=False, truncation=True, max_length=self.max_len)
        input_ids  = enc_full["input_ids"]
        n_prompt   = len(enc_prompt["input_ids"])
        labels = [-100] * n_prompt + input_ids[n_prompt:]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.ones(len(input_ids), dtype=torch.long), "labels": torch.tensor(labels)}

@dataclass
class LeftPadCausalCollator:
    tokenizer: object
    pad_to_multiple_of: Optional[int] = 8
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            rem = max_len % self.pad_to_multiple_of
            if rem: max_len += (self.pad_to_multiple_of - rem)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            L = len(f["input_ids"]); pad_left = max_len - L
            if pad_left > 0:
                ids  = torch.cat([torch.full((pad_left,), pad_id, dtype=torch.long), f["input_ids"]])
                attn = torch.cat([torch.zeros(pad_left, dtype=torch.long), f["attention_mask"]])
                labs = torch.cat([torch.full((pad_left,), -100, dtype=torch.long), f["labels"]])
            else:
                ids, attn, labs = f["input_ids"], f["attention_mask"], f["labels"]
            batch["input_ids"].append(ids); batch["attention_mask"].append(attn); batch["labels"].append(labs)
        return {k: torch.stack(v, dim=0) for k, v in batch.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="Data/vanilla_train.jsonl")
    ap.add_argument("--out_dir", default="Models/vanilla_sft")
    ap.add_argument("--per_device_batch", type=int, default=3)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    assert Path(args.train_jsonl).exists(), f"Missing {args.train_jsonl}. Run build_vanilla_sft.py first."

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    if torch.cuda.is_available(): model = model.to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    ds_train = SFTJsonl(args.train_jsonl, tok, args.max_len)
    collator = LeftPadCausalCollator(tok)

    targs = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        seed=args.seed,
        logging_strategy="steps", logging_steps=20, save_strategy="epoch", save_total_limit=2,
        disable_tqdm=False, report_to=[], bf16=torch.cuda.is_available(),
        gradient_checkpointing=True, optim="adamw_torch_fused", fp16=False,
        dataloader_num_workers=8, dataloader_pin_memory=True, dataloader_persistent_workers=True,
        remove_unused_columns=False, group_by_length=True,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds_train, data_collator=collator, tokenizer=tok)
    train_out = trainer.train()
    trainer.save_model(str(args.out_dir)); tok.save_pretrained(str(args.out_dir))
    print(train_out)

if __name__ == "__main__":
    main()
