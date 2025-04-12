import os
import subprocess
import uuid
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class RemoteExperiment(Enum):
    TRAIN_REWARD_MODEL = "train_reward_model"
    TRAIN_ENSEMBLE_REWARD_MODEL = "train_ensemble_reward_model"
    EVALUATE_REWARD_MODEL = "evaluate_reward_model"
    EVALUATE_ENSEMBLE_REWARD_MODEL = "evaluate_ensemble_reward_model"


JOBS_TEMPLATE = """#!/bin/sh
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/{job_name}/%j.out
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu
#SBATCH --gres=gpu:1

export HF_DATASETS_CACHE="/nas/ucb/dfeng/huggingface/datasets";

cd /nas/ucb/dfeng/code/reward-uncertainty;
eval "$(/nas/ucb/dfeng/anaconda3/bin/conda shell.bash hook)";
conda activate reward_uncertainty;

python -m experiments.{job_name} {job_args}
"""


class JobConfig(BaseModel):
    args: Optional[dict[str, str]] = None
    output_dir: str = os.getenv("SLURM_OUTPUT_DIR", "/nas/ucb/dfeng/slurm")
    mem_gb: int = 128
    time: str = "24:00:00"
    qos: str = "high"


def run_remote_experiment(job: RemoteExperiment, config: JobConfig, *, dry_run: bool = False, debug: bool = False):
    job_name = job.value
    job_args = " ".join([f"{k}={v}" for k, v in config.args.items()]) if config.args else ""
    job_template = JOBS_TEMPLATE.format(
        job_name=job_name,
        job_args=f"with {job_args}" if job_args else "",
        output_dir=config.output_dir,
        mem=f"{config.mem_gb}gb",
        time=config.time,
        qos=config.qos,
    )

    job_filename = f"{uuid.uuid4()}.sh"
    with open(job_filename, "w") as f:
        f.write(job_template)

    if debug:
        print("Submitting the following job:\n%s", job_template)

    if not dry_run:
        subprocess.run(f"sbatch {job_filename}", shell=True)

    os.remove(job_filename)
