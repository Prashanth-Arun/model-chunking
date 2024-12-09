#!/bin/bash

#SBATCH --job-name=latency_2_6_sft_test
#SBATCH --output=latency_2_6_sft.out
#SBATCH --error=latency_2_6_sft.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=t4v1,t4v2
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --qos=m4

export MKL_THREADING_LAYER=GNU
python -u -m scripts.test_latency_analysis
