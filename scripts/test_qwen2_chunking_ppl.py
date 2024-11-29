# ipython -i scripts/test_qwen2_chunking_ppl.py -- -v -b 32

from model_chunking.models.qwen2 import Qwen2ChunkingForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen2ChunkingConfig 
from tqdm.auto import tqdm
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import json
import os
import time

# Define function to calculate perplexity
def calculate_perplexity(model, tokenizer, dataset, batch_size):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    
    def process_batch(batch):
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        return outputs.loss.item()
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Perplexity", ncols=100):
        batch = dataset[i:i + batch_size]
        loss = process_batch(batch)
        total_loss += loss
        num_batches += 1

    avg_loss = total_loss / num_batches
    return math.exp(avg_loss)  # Perplexity is exp(loss)

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model1", "-m1", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Original model")
parser.add_argument("--model2", "-m2", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="New model")
parser.add_argument("--dataset", "-d", type=str, default="Salesforce/wikitext", help="Dataset")
parser.add_argument("--subset", "-s", type=str, default="wikitext-2-v1", help="Subset of the dataset")
parser.add_argument("--split", "-t", type=str, default="test", help="Split of the dataset")
parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size for evaluation")
parser.add_argument("--samples", "-n", type=int, default=-1, help="Number of samples to evaluate")
parser.add_argument("--do-variants", "-v", action="store_true", help="Evaluate variants of the model")
parser.add_argument("--chunking-mode", "-c", type=str, default="sequential", help="Chunking mode")
parser.add_argument("--aggregation-mode", "-a", type=str, default="mean", help="Aggregation mode")
args = parser.parse_args()

# Print arguments
print("Arguments:")
print(vars(args))

# Load dataset
dataset = load_dataset(args.dataset, args.subset)
if args.samples == -1:
    args.samples = len(dataset[args.split])
split_data = dataset[args.split].shuffle(seed=42).select(range(args.samples))

# Set Hugging Face dataset format to PyTorch for easier handling
split_data = split_data.with_format("torch")

# Load models
original_model = Qwen2ForCausalLM.from_pretrained(
    args.model1,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
original_tokenizer = AutoTokenizer.from_pretrained(args.model1)
original_config = Qwen2Config.from_pretrained(args.model1)

# Calculate perplexity for both models
print("Calculating perplexity for the original model...")
original_perplexity = calculate_perplexity(original_model, original_tokenizer, split_data, args.batch_size)
print(f"Original Model Perplexity ({args.model1}): {original_perplexity:.4f}")

if not args.do_variants:
    # NB: Change these details for an already trained chunking model
    config = Qwen2ChunkingConfig.from_pretrained(args.model2, num_layers_per_chunk=original_config.num_hidden_layers, chunking_mode=args.chunking_mode, aggregation_mode=args.aggregation_mode)
    new_model = Qwen2ChunkingForCausalLM.from_pretrained(
        args.model2,
        config=config,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    new_tokenizer = AutoTokenizer.from_pretrained(args.model2)

    print("Calculating perplexity for the new model...")
    new_perplexity = calculate_perplexity(new_model, new_tokenizer, split_data, args.batch_size)
    print(f"New Model Perplexity ({args.model2}): {new_perplexity:.4f}")

if args.do_variants:
    os.makedirs("./logs", exist_ok=True)
    f = open(f"./logs/test_qwen2_chunking_ppl_{int(time.time())}.txt", "w")
    divisors = [1, 2, 3, 4, 6, 8, 12, 24]
    assert original_config.num_hidden_layers == max(divisors), "Original model must have the same number of layers as the largest divisor"
    for divisor in tqdm(divisors):
        config = Qwen2ChunkingConfig.from_pretrained(args.model2, num_layers_per_chunk=original_config.num_hidden_layers // divisor, chunking_mode="sequential", aggregation_mode="mean")

        model = Qwen2ChunkingForCausalLM.from_pretrained(
            args.model2,
            config=config,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model2)

        print(f"Calculating perplexity for model with variant info: {args.model2}; num_layers_per_chunk={original_config.num_hidden_layers // divisor}; chunking_mode={args.chunking_mode}; aggregation_mode={args.aggregation_mode}")
        variant_ppl = calculate_perplexity(model, tokenizer, split_data, args.batch_size)

        print('-' * 20)
        print(f"Number of layers per chunk = {original_config.num_hidden_layers // divisor}")
        print(f"Perplexity: {variant_ppl:.4f}")
        print('-' * 20)
        print()

        f.write("-" * 20 + "\n")
        f.write(f"Number of layers per chunk = {original_config.num_hidden_layers // divisor}\n")
        f.write(f"Perplexity: {variant_ppl:.4f}\n")
        f.write("-" * 20 + "\n")
        f.write("\n")
        f.flush()

    f.close()