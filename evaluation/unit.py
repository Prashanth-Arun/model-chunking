from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass
from enum import Enum
from model_chunking import Qwen2ChunkingConfig, Qwen2ChunkingForCausalLM
from torch import Tensor
from typing import Dict, TypedDict
import json
import os


class DatasetID(Enum):
    WIKITEXT    = 0
    MAGPIE      = 1


@dataclass
class QwenWritableConfig(DataClassJsonMixin):
    experiment_name: str
    seed: int
    chunking_config: Dict
    dataset: DatasetID

    def __init__(
        self,
        experiment_name: str,
        seed: int,
        chunking_config: Qwen2ChunkingConfig|Dict,
        dataset: DatasetID
    ):
        self.experiment_name = experiment_name
        self.seed = seed
        if isinstance(chunking_config, Qwen2ChunkingConfig):
            self.chunking_config = chunking_config.to_dict()
        else:
            self.chunking_config = chunking_config
        self.dataset = dataset


@dataclass
class ExperimentResult(DataClassJsonMixin):
    latency: float
    perplexity: float


class MagpieData(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor


if __name__ == "__main__":

    experiment_config = QwenWritableConfig(
        experiment_name="test",
        chunking_config=Qwen2ChunkingConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"),
        dataset=DatasetID.WIKITEXT
    )

    with open(os.path.join(os.getcwd(), "config.json"), "w") as f:
        f.write(experiment_config.to_json())

    with open(os.path.join(os.getcwd(), "config.json"), "r") as f:
        exp_config_2 = QwenWritableConfig.from_json(f.read())

    print(f"\033[33m{experiment_config}\033[0m")
    print(f"\033[32m{exp_config_2}\033[0m")