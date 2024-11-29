from argparse import ArgumentParser
from datasets import load_dataset
from model_chunking import Qwen2ChunkingForCausalLM, Qwen2Tokenizer, Qwen2ChunkingConfig
from .unit import QwenWritableConfig, ExperimentResult, DatasetID
from tqdm import tqdm
from time import time
from typing import Dict
import os
import torch

# Evaluation Constants
MAX_LENGTH = 4096
STRIDE_LENGTH = 256


def evaluate(
    model: Qwen2ChunkingForCausalLM,
    dataset: DatasetID,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
) -> tuple[float, float]:
    
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
    
    match dataset:
        case DatasetID.WIKITEXT:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="validation")
        case DatasetID.MAGPIE:
            dataset = None # TODO: Change this (Would Eval be different for Magpie?)
        case _:
            raise ValueError("Invalid dataset specified for evaluation")
        
    assert dataset is not None
    encoded_inputs = tokenizer("\n\n".join(dataset['text']), return_tensors="pt")
    sequence_length = encoded_inputs['input_ids'].shape[-1]

    losses : list[float] = []
    latency : list[float] = []
    last_end = 0

    for start in tqdm(range(0, sequence_length, STRIDE_LENGTH)):
        end = min(start + MAX_LENGTH, sequence_length)
        target_length = end - last_end
        input_ids = encoded_inputs['input_ids'][:, start : end].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-target_length] = -100

        with torch.no_grad():
            commenced = time()
            outputs = model(input_ids, labels=target_ids)
            concluded = time()
            loss = outputs.loss

        losses.append(loss)
        latency.append(concluded - commenced)
        last_end = end
        if end == sequence_length: break

        del target_ids, input_ids

    perplexity_score = torch.exp(torch.stack(losses).mean())
    average_latency = sum(latency) / len(latency)
    return average_latency, perplexity_score.item()



def execute(
    config: QwenWritableConfig
) -> ExperimentResult:

    model_config = Qwen2ChunkingConfig.from_dict(config.chunking_config)
    model = Qwen2ChunkingForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        config=model_config,
        torch_dtype="auto",
        device_map="auto",
    )
    latency, perplexity = evaluate(
        model=model,
        dataset=config.dataset
    )
    return ExperimentResult(
        latency=latency,
        perplexity=perplexity
    )


if __name__ == "__main__":
    
    # Read in the arguments
    parser = ArgumentParser()
    parser.add_argument("base_path", type=str)
    args = parser.parse_args()
    assert isinstance(args.base_path, str)
    exp_path = args.base_path

    # Load in the config file
    config_path = os.path.join(exp_path, "config.json")
    assert os.path.exists(config_path), f"ERROR: config file does not exist at location {config_path}"

    with open(config_path, "r") as f:
        config = QwenWritableConfig.from_json(f.read())

    # Execute the training_process
    results = execute(config=config)

    # Write the results to a JSON file
    results_path = os.path.join(exp_path, "results.json")
    with open(results_path, "w") as f:
        f.write(results.to_json())