#!/usr/bin/env python
"""
STaR Bootstrapping Script (strict-match version + smart resampling)
-------------------------------------------------------------------
Adds controlled resampling for remaining hard examples with small temperature.
Only bootstraps examples where the predicted final answer EXACTLY matches the gold answer.
"""

import os, sys, re, json, time, math, argparse
from pathlib import Path
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# make local imports available
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from shared import (
    prompt_cot, prompt_with_hint, gsm8k_extract_gold,
    extract_pred, build_vllm_engine, vllm_generate, MODEL_NAME
)

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def strip_control_tokens(t: str):
    t = t.replace("<<SYS>>", "").replace("<</SYS>>", "")
    t = re.sub(r"\[/?INST\]", "", t)
    return t

def clean_rationale_text(text: str):
    """Remove instruction tokens and trailing final line."""
    t = strip_control_tokens(text or "").strip()
    m = re.search(r"####\s*-?\d[\d,]*\.?\d*", t)
    if m:
        t = t[:m.start()]
    return t.rstrip()

def ensure_final_hash(text, gold):
    """Make sure rationale ends with the correct #### gold line."""
    t = text.rstrip()
    if not re.search(rf"####\s*{re.escape(gold)}\s*$", t):
        t += f"\n#### {gold}"
    return t

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=MODEL_NAME)
    ap.add_argument("--out", default="Data/star_train_from_base.jsonl")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=640)
    ap.add_argument("--max_passes", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--engine", choices=["hf", "vllm"], default="vllm")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--resample_tries", type=int, default=2)
    ap.add_argument("--resample_temperature", type=float, default=0.25)
    ap.add_argument("--resample_top_p", type=float, default=0.95)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[warn] Removing existing {out}")
        out.unlink()

    # ---------------------------------------------------------
    # Engine setup
    # ---------------------------------------------------------
    if args.engine == "hf":
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=dtype, device_map="auto"
        ).eval()
        print("Engine: HF")

        def generate(prompts, temperature=0.0, top_p=1.0):
            do_sample = temperature > 0.0
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            gen_cfg = GenerationConfig(
                do_sample=do_sample, top_p=top_p, temperature=temperature,
                max_new_tokens=args.max_new,
                bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
            )
            with torch.inference_mode():
                out_ids = model.generate(**inputs, generation_config=gen_cfg)
            return tok.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    else:
        tok = None
        engine = build_vllm_engine(args.model_name, tensor_parallel_size=args.tensor_parallel_size)
        print("Engine: vLLM (eager)")

        def generate(prompts, temperature=0.0, top_p=1.0):
            return vllm_generate(engine, prompts, max_new_tokens=args.max_new, temperature=temperature, top_p=top_p)

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    train = load_dataset("gsm8k", "main", split="train")
    N = len(train)
    qs = [train[i]["question"].strip() for i in range(N)]
    golds = [gsm8k_extract_gold(train[i]["answer"]) for i in range(N)]
    filled = [False] * N
    answers = [None] * N
    self_ok_total = 0
    hint_ok_total = 0

    # ---------------------------------------------------------
    # Outer loop rounds
    # ---------------------------------------------------------
    t0 = time.time()
    for round_idx in range(args.rounds):
        print(f"\n=== STaR Round {round_idx+1}/{args.rounds} ===")
        t_round = time.time()

        for pass_idx in range(args.max_passes):
            remaining = [i for i in range(N) if not filled[i] and golds[i] is not None]
            if not remaining:
                break
            using_hint = (pass_idx > 0)
            desc = f"pass {pass_idx+1}/{args.max_passes} ({'hint' if using_hint else 'self'})"
            for s in tqdm(
                range(0, len(remaining), args.batch),
                total=math.ceil(len(remaining)/args.batch),
                desc=desc, dynamic_ncols=True
            ):
                idxs = remaining[s:s+args.batch]
                q_batch = [qs[i] for i in idxs]
                g_batch = [golds[i] for i in idxs]
                prompts = (
                    [prompt_with_hint(q, g) for q, g in zip(q_batch, g_batch)]
                    if using_hint else [prompt_cot(q) for q in q_batch]
                )
                decodes = generate(prompts)

                for local_k, i_idx in enumerate(idxs):
                    if filled[i_idx]:
                        continue
                    dec = decodes[local_k]
                    gold = g_batch[local_k]
                    pred = extract_pred(dec)

                    if pred == gold:
                        rationale = clean_rationale_text(dec)
                        answers[i_idx] = ensure_final_hash(rationale, gold)
                        filled[i_idx] = True
                        if using_hint:
                            hint_ok_total += 1
                        else:
                            self_ok_total += 1
                    elif args.resample_tries > 0:
                        # Retry generation for hard items
                        for attempt in range(args.resample_tries):
                            redec = generate(
                                [prompts[local_k]],
                                temperature=args.resample_temperature,
                                top_p=args.resample_top_p
                            )[0]
                            repred = extract_pred(redec)
                            if repred == gold:
                                rationale = clean_rationale_text(redec)
                                answers[i_idx] = ensure_final_hash(rationale, gold)
                                filled[i_idx] = True
                                if using_hint:
                                    hint_ok_total += 1
                                else:
                                    self_ok_total += 1
                                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        round_cov = sum(filled)
        print(f"[round {round_idx+1}] coverage={round_cov}/{N} "
              f"({round_cov/N:.1%}) | self_ok={self_ok_total} | hint_ok={hint_ok_total} "
              f"| time={(time.time()-t_round)/60:.1f}min")

    elapsed = time.time() - t0
    print(f"\nFilled {sum(filled)}/{N} in {elapsed/60:.1f} min "
          f"| self_ok={self_ok_total} | hint_ok={hint_ok_total}")

    # ---------------------------------------------------------
    # Write final JSONL
    # ---------------------------------------------------------
    kept = 0
    with open(out, "w", encoding="utf-8") as f:
        for i in range(N):
            if filled[i] and answers[i] is not None:
                rec = {
                    "question": qs[i],
                    "rationale": clean_rationale_text(answers[i]),
                    "answer": golds[i]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Wrote {kept} examples to {out}")
    if kept == N:
        print("✅ STaR bootstrap complete (all train items correct)")
    else:
        print("⚠️ STaR bootstrap partial; consider raising --rounds or --max_passes")


if __name__ == "__main__":
    main()
