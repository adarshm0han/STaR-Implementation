#!/usr/bin/env python
import os, sys, re, json, time, math, argparse
from pathlib import Path
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from shared import build_vllm_engine, vllm_generate

def extract_gold(ans_text: str):
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", ans_text.strip()); return m.group(1) if m else None
def extract_pred(text: str):
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", text); return m[-1] if m else None
def exact_match(a, b):
    if a is None or b is None: return False
    return a.replace(",", "").strip() == b.replace(",", "").strip()

def build_prompt(q: str) -> str:
    return (
        "<s>[INST] <<SYS>>\n"
        "You are a careful math tutor. Solve step by step. End with: #### <number>\n"
        "<</SYS>>\n"
        f"Q: {q.strip()}\nA: [/INST]"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="HF model dir, e.g., Models/vanilla_sft or Models/star_sft")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--engine", choices=["hf","vllm"], default="vllm")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    pred_path    = Path(args.save_dir) / "preds.jsonl"
    metrics_path = Path(args.save_dir) / "metrics.json"

    ds = load_dataset("gsm8k", "main", split="test")
    if args.limit is not None: ds = ds.select(range(args.limit))
    N = len(ds)

    if args.engine == "hf":
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        tok.pad_token = tok.eos_token; tok.padding_side = "left"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype, device_map="auto").eval()
        print("Engine: HF")
        def generate(prompts):
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            gen_cfg = GenerationConfig(do_sample=False, top_p=1.0, max_new_tokens=args.max_new,
                                       eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            with torch.inference_mode():
                out = model.generate(**inputs, generation_config=gen_cfg)
            return tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    else:
        print("Engine: vLLM")
        engine = build_vllm_engine(args.model_dir, tensor_parallel_size=args.tensor_parallel_size)
        def generate(prompts):
            return vllm_generate(engine, prompts, max_new_tokens=args.max_new, temperature=0.0, top_p=1.0)

    rows, correct = [], 0
    t0 = time.time()
    for i in tqdm(range(0, N, args.batch), total=math.ceil(N/args.batch), desc="Eval", dynamic_ncols=True):
        batch = ds.select(range(i, min(i+args.batch, N)))
        prompts = [build_prompt(q) for q in batch["question"]]
        decodes = generate(prompts)
        for q, gold_ans, gen in zip(batch["question"], batch["answer"], decodes):
            gold = extract_gold(gold_ans); pred = extract_pred(gen)
            ok = exact_match(pred, gold); correct += int(ok)
            rows.append({"question": q, "gold": gold, "prediction": pred, "generation": gen, "correct": ok})

    acc = correct / N
    elapsed = time.time() - t0
    print(f"\nExact-match: {acc:.4f} | N={N} | time={elapsed:.1f}s")

    with open(pred_path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"method":"eval","n":N,"correct":correct,"accuracy_exact_match":acc,"engine":args.engine}, f, indent=2)
    print("Saved:", pred_path, metrics_path)

if __name__ == "__main__":
    main()
