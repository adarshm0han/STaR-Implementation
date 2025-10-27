Implemented the STaR (Self-Taught Reasoner) framework to bootstrap chain-of-thought (CoT)
 reasoning on GSM8K using meta-llama/Llama-3.2-3B-Instruct. Following the assignment constraints,
 we compare three methods: (i) Zero-Shot CoT prompting, (ii) Vanilla SFT (fine-tuning on GSM8K
 rationales from the train split only), and (iii) STaR (iterative rationale generation with rationalization
 hints and SFT). Using consistent decoding with vLLM, we obtain exact-match (EM) accuracies on
 GSM8K test of 46.78% (Zero-Shot CoT), 65.81% (Vanilla SFT), and 78.17% (STaR). We include full
 prompts, workflow, commands, results, and analysis
