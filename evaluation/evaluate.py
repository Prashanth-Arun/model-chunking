from argparse import ArgumentParser
from datasets import load_dataset, arrow_dataset
from model_chunking import Qwen2ChunkingForCausalLM, Qwen2Tokenizer, Qwen2ChunkingConfig
from .unit import QwenWritableConfig, ExperimentResult, DatasetID, MagpieData
from tqdm import tqdm
from time import time
from transformers import BatchEncoding
from typing import Dict
import os
import torch

# Evaluation Constants for WikiText
MAX_LENGTH = 4096
STRIDE_LENGTH = 256

# Tokenization Template for Magpie-Instruction
MESSAGE_TEMPLATE = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": None},
    {"role": "assistant", "content": None}
]

# Helper(s) for Magpie Evaluation

def filter_inputs_by_level(
    dataset : arrow_dataset.Dataset,
    level : str = 'easy'
) -> list[int]:
    indices : list[int] = []
    for i, example in enumerate(dataset):
        if example['difficulty'] == level:
            indices.append(i)
    return indices


def get_chats(
    dataset : arrow_dataset.Dataset,
    tokenizer : Qwen2Tokenizer
) -> list[tuple[str, int]]:
    datapoints : list[tuple[str, int]] = []

    def find_sublist_end(lst : list[str], sublist : list[str]) -> int:
        for i in range(len(lst) - len(sublist) + 1):
            if lst[i:i+len(sublist)] == sublist:
                return i + len(sublist)
        return -1
    
    def get_prompt(instruction : str, response : str):
        datapoint = MESSAGE_TEMPLATE.copy()
        datapoint[1]['content'] = instruction
        datapoint[2]['content'] = response
        template = tokenizer.apply_chat_template(
            datapoint,
            tokenize=False,
            add_generation_prompt=False
        )
        return template
    
    def find_response_start(chat_template : str) -> int:
        tokenized_chat_template = tokenizer.tokenize(chat_template)
        return find_sublist_end(tokenized_chat_template, ['<|im_start|>', 'assistant']) + 1

    for example in dataset:
        chat_template = get_prompt(example['instruction'], example['response'])
        response_start = find_response_start(chat_template)
        datapoints.append(tuple([chat_template, response_start]))

    return datapoints


def get_model_inputs(chat : str, response_start : int, tokenizer : Qwen2Tokenizer) -> MagpieData:
    encoding = tokenizer(chat, return_tensors='pt')
    labels = encoding['input_ids'].clone()
    labels[0, : response_start] = -100 
    return MagpieData(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        labels=labels
    )


# Main Evaluation Functions

def evaluate_wikitext(
    model: Qwen2ChunkingForCausalLM,
    tokenizer: Qwen2Tokenizer,
) -> tuple[float, float]:
    
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="validation")
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

    return latency, losses


def evaluate_magpie(
    model: Qwen2ChunkingForCausalLM,
    tokenizer: Qwen2Tokenizer,
    limit: int = 500,
    task_level : str = "easy"
) -> tuple[float, float]:
    
    dataset = load_dataset("Magpie-Align/Magpie-Qwen2-Pro-200K-English")['train']
    filtered_dataset_indices = filter_inputs_by_level(dataset, level=task_level)
    dataset = dataset.select(filtered_dataset_indices[: min(limit, len(filtered_dataset_indices))])
    chats = get_chats(dataset=dataset, tokenizer=tokenizer)

    losses : list[float] = []
    latency : list[float] = []

    for i, chat in enumerate(chats):
        template, response_start = chat
        datapoint = get_model_inputs(template, response_start, tokenizer)
        datapoint = { k : v.to(model.device) for k, v in datapoint.items() }
        start = time()
        with torch.no_grad():
            model_out = model(
                input_ids=datapoint['input_ids'],
                attention_mask=datapoint['attention_mask'],
                labels=datapoint['labels']
            )
        end = time()
        normalized_latency = (end - start) / datapoint['labels'].shape[-1]
        losses.append(model_out['loss'])
        latency.append(normalized_latency)
        print(i, torch.cuda.memory_allocated(), datapoint['labels'].shape)
        datapoint = { k : v.cpu() for k, v in datapoint.items() }

        del datapoint, model_out

    return latency, losses


def evaluate(
    model: Qwen2ChunkingForCausalLM,
    dataset: DatasetID,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
) -> tuple[float, float]:
    
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
    
    match dataset:
        case DatasetID.WIKITEXT:
            latency, losses = evaluate_wikitext(model=model, tokenizer=tokenizer)
        case DatasetID.MAGPIE:
            latency, losses = evaluate_magpie(model=model, tokenizer=tokenizer)
        case _:
            raise ValueError("Invalid dataset specified for evaluation")

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

    # Set the seed for this experiment
    torch.manual_seed(config.seed)

    # Execute the training_process
    results = execute(config=config)

    # Write the results to a JSON file
    results_path = os.path.join(exp_path, "results.json")
    with open(results_path, "w") as f:
        f.write(results.to_json())