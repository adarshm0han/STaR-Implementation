# STaR-Implementation
<img width="1434" height="980" alt="diagram-export-11-4-2025-12_31_17-AM" src="https://github.com/user-attachments/assets/cd6d7e97-a3b1-4a1c-bf14-eca77571bfec" />

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


## Results

We report exact-match (EM) accuracy on GSM8K test following consistent decoding settings (vLLM):

| Method | EM Accuracy |
|--------|-------------:|
| Zero-Shot CoT | 46.78% |
| Vanilla SFT | 65.81% |
| STaR (iterative) | 78.17% |

Interpretation: STaR provides substantial gains over both prompting and vanilla supervised fine-tuning by iteratively producing and refining intermediate rationales and using them in SFT.

## High-level workflow

1. Pull GSM8K dataset (train/test) with questions, answers and optional gold rationales.
2. Evaluate a base instruction-tuned model (meta-llama/Llama-3.2-3B-Instruct) with Zero-Shot CoT prompts to obtain a baseline.
3. Train a Vanilla SFT model on the train split rationales to get the SFT baseline.
4. Run STaR iterations:
   - Use the current model to generate rationales for training examples with just question and gold answer.
   - Bootstrap these rationales to the dataset only if the answer is correct.
   - Retrain previous iteration's model on this bootstrapped dataset.
   - Repeat the bootstrapping  and retraining until all examples are appended with the correct rationales.
   - Save the final model.
5. Evaluate all models on the GSM8K test split using consistent decoding to compute EM accuracy.

## Reproducing the experiments (high-level)

Prerequisites:

- Python 3.11 (recommended)
- CUDA-enabled GPU for efficient training and inference
- vLLM for bulk inference/bootstrapping
- Typical ML stack: torch, transformers, datasets, and other utilities (see requirements.txt if present)

High-level commands : Please check the report.

Notes: the repository may contain scripts and prompt files. Replace script names and paths with the actual files in this repo.

## Prompts and configuration

- All prompts used for Zero-Shot CoT, rationalization hints, and STaR generation are included in this repository (see prompts/). If prompts are not present, check the notebooks or experiment notes.
- Use deterministic decoding (e.g., greedy or fixed sampling with seed) and document decoding settings to ensure reproducibility.

## Evaluation

- We use exact-match (EM) accuracy on GSM8K test as the primary metric. Where possible, also report answer-token match, and analyze common failure modes.


## Contributing

Contributions, issue reports, and pull requests are welcome. Please open an issue to discuss larger changes before submitting a PR.

## References

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. STaR: Self-Taught Rea
soner—Bootstrapping Reasoning With Reasoning. arXiv:2203.14465, 2022.

## Contact

For questions or collaboration, open an issue or reach out to the repository owner: @adarshm0han.
Linkedin: https://www.linkedin.com/in/adarshm0han/
