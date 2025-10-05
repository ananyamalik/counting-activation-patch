#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=benchmark_10k
#SBATCH --output=logs/benchmark_10k_%j.log
#SBATCH --error=logs/benchmark_10k_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load anaconda3/2024.06

source ~/.bashrc
conda activate emotional-safety

python3 run_benchmarks.py --num 10000 --model "meta-llama/Meta-Llama-3-8B-Instruct" > output_10k.txt
