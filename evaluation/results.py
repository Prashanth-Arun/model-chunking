
from .unit import ExperimentResult, QwenWritableConfig
import os

BASE_PATH = os.path.join(os.getcwd(), ".experiments")

def load_results(
    config: QwenWritableConfig,
    base_path: str = BASE_PATH
) -> ExperimentResult:
    
    exp_path = os.path.join(base_path, config.experiment_name)
    if not os.path.exists(exp_path):
        raise LookupError(f"ERROR: Can't find experiment at {exp_path}")
    
    results_path = os.path.join(exp_path, "results.json")
    if not os.path.exists(results_path):
        raise LookupError(f"ERROR: Can't find results file at {results_path}")

    with open(results_path, "r") as f:
        result = ExperimentResult.from_json(f.read())

    return result