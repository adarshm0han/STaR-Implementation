#!/usr/bin/env python
import os, sys, re, json, time, math, argparse
from pathlib import Path
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import GenerationConfig

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from shared import load_base_model_full, prompt_cot, extract_pred, gsm8k_extract_gold, build_vllm_engine, vllm_generate

def exact_match(a, b):
    if a is None or b is None: return False
    return a.replace(",", "").strip() == b.replace(",", "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="Data/eval_zeroshot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--engine", choices=["hf","vllm"], default="vllm")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    pred_path    = Path(args.save_dir) / "zeroshot_preds.jsonl"
    metrics_path = Path(args.save_dir) / "zeroshot_metrics.json"

    ds = load_dataset("gsm8k", "main", split="test")
    if args.limit is not None:
        ds = ds.select(range(args.limit))
    N = len(ds)
    print("Eval N:", N)

    if args.engine == "hf":
        tok, model = load_base_model_full()
        print("Engine: HF | Device:", model.device)
        def generate(prompts):
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            gen_cfg = GenerationConfig(
                max_new_tokens=args.max_new,
                do_sample=False, top_p=1.0,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            )
            with torch.inference_mode():
                out = model.generate(**inputs, generation_config=gen_cfg)
            return tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    else:
        engine = build_vllm_engine(tensor_parallel_size=args.tensor_parallel_size)
        print("Engine: vLLM")
        def generate(prompts):
            return vllm_generate(engine, prompts, max_new_tokens=args.max_new, temperature=0.0, top_p=1.0)

    rows, correct = [], 0
    t0 = time.time()
    for i in tqdm(range(0, N, args.batch_size), total=math.ceil(N/args.batch_size), desc="Zero-Shot CoT", dynamic_ncols=True):
        batch = ds.select(range(i, min(i+args.batch_size, N)))
        prompts = [prompt_cot(q.strip()) for q in batch["question"]]
        decodes = generate(prompts)
        for q, gold_ans, gen_text in zip(batch["question"], batch["answer"], decodes):
            gold = gsm8k_extract_gold(gold_ans)
            pred = extract_pred(gen_text)
            ok   = exact_match(pred, gold)
            correct += int(ok)
            rows.append({"question": q, "gold": gold, "prediction": pred, "generation": gen_text, "correct": ok})

    acc = correct / N
    elapsed = time.time() - t0
    print(f"Zero-Shot CoT exact-match: {acc:.4f} | N={N} | time={elapsed:.1f}s")

    with open(pred_path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"method":"zero_shot_cot","n":N,"correct":correct,"accuracy_exact_match":acc,
                   "batch_size":args.batch_size,"max_new_tokens":args.max_new,"engine":args.engine}, f, indent=2)
    print("Saved predictions to:", pred_path)
    print("Saved metrics to:", metrics_path)

if __name__ == "__main__":
    main()
