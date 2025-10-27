# STaR-Implementation

Implementation of the STaR (Self-Taught Reasoner) framework to bootstrap chain-of-thought (CoT) reasoning on GSM8K using meta-llama/Llama-3.2-3B-Instruct.

## Summary

This repository implements and compares three approaches for improving chain-of-thought reasoning on GSM8K:

- Zero-Shot CoT prompting
- Vanilla Supervised Fine-Tuning (SFT) on GSM8K rationales
- STaR: iterative rationale generation + rationalization hints + SFT

Using consistent decoding with vLLM, the reported exact-match (EM) accuracies on the GSM8K test set are:

- Zero-Shot CoT: 46.78%
- Vanilla SFT: 65.81%
- STaR: 78.17%

The repo contains prompts, workflow notes, commands, results, and analysis to reproduce or extend the experiments.

## Table of contents

- Summary
- Results
- High-level workflow
- Reproducing the experiments
- Prompts and configuration
- Evaluation
- Contributing
- License
- Contact

## Results

We report exact-match (EM) accuracy on GSM8K test following consistent decoding settings (vLLM):

| Method | EM Accuracy |
|--------|-------------:|
| Zero-Shot CoT | 46.78% |
| Vanilla SFT | 65.81% |
| STaR (iterative) | 78.17% |

Interpretation: STaR provides substantial gains over both prompting and vanilla supervised fine-tuning by iteratively producing and refining intermediate rationales and using them in SFT.

## High-level workflow

1. Prepare GSM8K dataset (train/validation/test) with questions and optional gold rationales.
2. Evaluate a base instruction-tuned model (meta-llama/Llama-3.2-3B-Instruct) with Zero-Shot CoT prompts to obtain a baseline.
3. Train a Vanilla SFT model on the train split rationales to get the SFT baseline.
4. Run STaR iterations:
   - Use the current model to generate rationales for training examples, optionally using rationalization hints.
   - Curate / filter the generated rationales (heuristics or automatic scoring).
   - Fine-tune (SFT) on the curated rationales to obtain an improved model.
   - Repeat the generate-filter-SFT loop for several iterations until convergence or performance plateau.
5. Evaluate all models on the GSM8K test split using consistent decoding to compute EM accuracy.

## Reproducing the experiments (high-level)

Prerequisites:

- Python 3.8+ (recommended)
- CUDA-enabled GPU for efficient training and inference (if using local GPUs)
- vLLM for consistent decoding (or an alternate deterministic decoding setup)
- Typical ML stack: torch, transformers, datasets, and other utilities (see requirements.txt if present)

High-level commands (replace placeholders as needed):

1. Install dependencies:

   pip install -r requirements.txt

2. Prepare GSM8K dataset (using Hugging Face datasets or provided scripts):

   python scripts/prepare_gsm8k.py --out data/gsm8k

3. Evaluate Zero-Shot CoT:

   python eval_zero_shot.py --model meta-llama/Llama-3.2-3B-Instruct --data data/gsm8k/test --prompt prompts/zero_shot_cot.md

4. Train Vanilla SFT:

   python train_sft.py --model meta-llama/Llama-3.2-3B-Instruct --train data/gsm8k/train --val data/gsm8k/val --output models/vanilla_sft

5. Run STaR iterations (example loop):

   for iter in 1 2 3; do
     python generate_rationales.py --model models/current --data data/gsm8k/train --out data/rationales/iter${iter}.jsonl --hints prompts/rationalization_hints.md
     python filter_rationales.py --input data/rationales/iter${iter}.jsonl --output data/curated/iter${iter}.jsonl
     python train_sft.py --model meta-llama/Llama-3.2-3B-Instruct --train data/curated/iter${iter}.jsonl --val data/gsm8k/val --output models/iter${iter}
     # set models/current to models/iter${iter}
   done

Notes: the repository may contain scripts and prompt files. Replace script names and paths with the actual files in this repo.

## Prompts and configuration

- All prompts used for Zero-Shot CoT, rationalization hints, and STaR generation are included in this repository (see prompts/). If prompts are not present, check the notebooks or experiment notes.
- Use deterministic decoding (e.g., greedy or fixed sampling with seed) and document decoding settings to ensure reproducibility.

## Evaluation

- We use exact-match (EM) accuracy on GSM8K test as the primary metric. Where possible, also report answer-token match, and analyze common failure modes.
- When reporting numbers, include decoding parameters and seeds.

## Contributing

Contributions, issue reports, and pull requests are welcome. Please open an issue to discuss larger changes before submitting a PR.

## License

This repository does not include a license file by default. Add a LICENSE file to clarify usage (e.g., MIT, Apache-2.0).

## Contact

For questions or collaboration, open an issue or reach out to the repository owner: @adarshm0han
