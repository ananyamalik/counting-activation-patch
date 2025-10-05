# 🧪 Language Model Zero-Shot Benchmark

This repository contains the benchmarking and activation patching experiments to see whether open-weighted LLMs are able to count the number of category specific items in a lsit
---
## Datasets

Files: ['diverse_counts.json', 'diverse_counts_with_running.json', 'create_data.py]

## Benchmark Results

Files: ['model.py', 'compile_benchmark.ipynb']

| Model                        | Accuracy |
|------------------------------|-----------|
| Llama-2-7b-hf                | 0.1089 |
| Mistral-7B-Instruct-v0.2     | 0.2716 |
| Meta-Llama-3-8B-Instruct     | 0.5062 |

Meta-Llama-3-8B-Instruct achieved the highest zero-shot accuracy on this task.

---

## Activation Patching

Files: ['mediation_with_activation_patching.py', 'model_patch.ipynb']