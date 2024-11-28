from .unit import QwenWritableConfig
import os
import subprocess

BASE_PATH = os.path.join(os.getcwd(), ".experiments")

def run_slurm(
    config: QwenWritableConfig,
    base_path: str = BASE_PATH
):
    experiment_name = config.experiment_name
    exp_path = os.path.join(base_path, experiment_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    with open(os.path.join(exp_path, "config.json"), "w") as f:
        f.write(config.to_json())

    module = f"python -u -m evaluation.evaluate {exp_path}"
    output_path = os.path.join(exp_path, "slurm.out")
    error_path = os.path.join(exp_path, "slurm.err")

    sbatch_args = [
        f"#SBATCH --job-name={experiment_name}",
        f"#SBATCH --output={output_path}",
        f"#SBATCH --error={error_path}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=16G",
        "#SBATCH --partition=t4v1,t4v2",
        "#SBATCH --time=01:00:00",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --qos=m5",
    ]

    job_path = os.path.join(exp_path, "job.sh")
    with open(job_path, "w") as jf:
        jf.writelines(f"{line}\n" for line in [
            "#!/bin/bash",
            "",
            *sbatch_args,
            "",
            "export MKL_THREADING_LAYER=GNU",
            module
        ])

    process_result = subprocess.run(
        args=["sbatch", job_path],
        capture_output=True
    )
    return_code = process_result.returncode
    if return_code != 0:
        print(f"ERROR: Failed to submit slurm job {experiment_name}. Reason: {process_result.stderr.decode()}")
    else:
        print(f"{process_result.stdout.decode()} for experiment {experiment_name}")

    