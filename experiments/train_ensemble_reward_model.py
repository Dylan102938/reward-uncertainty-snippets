import json
import os
import random
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from dotenv import load_dotenv
from sacred import Experiment
from sacred.observers import FileStorageObserver

from experiments.train_reward_model import get_tokenizer
from experiments.train_reward_model import sacred_config as train_rm_config
from experiments.utils.data_processing.datasets import (
    HHRLHFPreprocessor,
    get_hh_rlhf_dataset,
)
from experiments.utils.misc import get_updated_fields
from reward_uncertainty.model.ensemble.simple_ensemble import (
    SimplePreferenceModelEnsemble,
)
from reward_uncertainty.model.simple_preference_model import SimplePreferenceModel
from reward_uncertainty.store.disk_store import DiskBackingStore
from reward_uncertainty.types import DataSubset

load_dotenv()
ex = Experiment("train_ensemble")
access_token = os.getenv("HF_ACCESS_TOKEN") or ""
backing_store = DiskBackingStore.create(SimplePreferenceModelEnsemble.generate_id())


@ex.config
def sacred_config():
    num_models: int = 2  # noqa: F841
    use_bootstrapped_dataset: bool = False  # noqa: F841
    shuffle_train_dataset: bool = False  # noqa: F841
    randomly_sample_params: bool = False  # noqa: F841


ex.add_config(train_rm_config())


def preprocess_dataset(dataset: Dataset, tokenizer_name: str, max_length: int) -> Dataset:
    original_columns = dataset.column_names
    tokenizer = get_tokenizer(tokenizer_name, access_token)
    num_proc = 24
    dataset = dataset.map(
        HHRLHFPreprocessor(tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    return dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
    )


@ex.main
def main(
    num_models: int,
    tokenizer_name: Optional[str],
    model_name: str,
    max_length: int,
    data_subset: DataSubset,
    data_path: str,
    use_bootstrapped_dataset: bool,
    shuffle_train_dataset: bool,
    randomly_sample_params: bool,
):
    assert ex.current_run is not None, "This script must be run with Sacred"
    print("Starting Experiment: train_ensemble")

    torch.set_anomaly_enabled(True)

    tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name

    train_dataset = get_hh_rlhf_dataset(data_subset, "train", data_path)
    train_dataset = preprocess_dataset(train_dataset, tokenizer_name, max_length)

    eval_dataset = get_hh_rlhf_dataset(data_subset, "test", data_path)
    eval_dataset = preprocess_dataset(eval_dataset, tokenizer_name, max_length)

    learning_rates = [np.random.uniform(5e-7, 3e-5) for _ in range(num_models)] if randomly_sample_params else None
    num_train_epochs = [np.random.choice([1, 2]) for _ in range(num_models)] if randomly_sample_params else None
    lora_ranks = [random.choice([8, 16, 32, 64]) for _ in range(num_models)] if randomly_sample_params else None
    lora_alphas = [random.choice([32, 64, 128]) for _ in range(num_models)] if randomly_sample_params else None
    lora_dropouts = (
        [random.choice([0.0, 0.05, 0.1, 0.15]) for _ in range(num_models)] if randomly_sample_params else None
    )

    if randomly_sample_params:
        print("====== Randomly Sampled Parameters ======")
        for i in range(num_models):
            model_params = {
                "learning_rate": learning_rates[i],
                "num_train_epochs": num_train_epochs[i],
                "lora_rank": lora_ranks[i],
                "lora_alpha": lora_alphas[i],
                "lora_dropout": lora_dropouts[i],
            }
            print(f"Model {i + 1}: {json.dumps(model_params, indent=2)}")

    updated_config = SimplePreferenceModelEnsemble.TrainingConfig(
        access_token=access_token,
        train_dataset_name=f"hh_rlhf_{data_subset}",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_ensemble_members=num_models,
        bootstrap_train_dataset=use_bootstrapped_dataset,
        shuffle_train_dataset=shuffle_train_dataset,
        learning_rate=ex.current_run.config["learning_rate"],
        learning_rate_fn=(lambda i: learning_rates[i]) if learning_rates else None,
        lr_scheduler_type=ex.current_run.config["lr_scheduler_type"],
        num_train_epochs=ex.current_run.config["num_train_epochs"],
        num_train_epochs_fn=((lambda i: num_train_epochs[i]) if num_train_epochs else None),
        per_device_train_batch_size=ex.current_run.config["per_device_train_batch_size"],
        weight_decay=ex.current_run.config["weight_decay"],
        model_name=ex.current_run.config["model_name"],
        entropy_coeff=ex.current_run.config["entropy_coeff"],
        gradient_accumulation_steps=ex.current_run.config["gradient_accumulation_steps"],
        gradient_checkpointing=ex.current_run.config["gradient_checkpointing"],
        deepspeed=ex.current_run.config["deepspeed"],
        local_rank=ex.current_run.config["local_rank"],
        bf16=ex.current_run.config["bf16"],
        optim=ex.current_run.config["optim"],
        tokenizer_name=ex.current_run.config["tokenizer_name"],
        max_length=ex.current_run.config["max_length"],
        resume_from_checkpoint=ex.current_run.config["resume_from_checkpoint"],
        lora_rank=ex.current_run.config["lora_rank"],
        lora_rank_fn=(lambda i: lora_ranks[i]) if lora_ranks else None,
        lora_alpha=ex.current_run.config["lora_alpha"],
        lora_alpha_fn=(lambda i: lora_alphas[i]) if lora_alphas else None,
        lora_dropout=ex.current_run.config["lora_dropout"],
        lora_dropout_fn=(lambda i: lora_dropouts[i]) if lora_dropouts else None,
    )

    print("====== Updated Preference Model Config ======")
    for i in range(num_models):
        updated_simple_config = {
            **updated_config.model_dump(),
            **({"learning_rate": learning_rates[i]} if learning_rates else {}),
            **({"num_train_epochs": num_train_epochs[i]} if num_train_epochs else {}),
            **({"lora_rank": lora_ranks[i]} if lora_ranks else {}),
            **({"lora_alpha": lora_alphas[i]} if lora_alphas else {}),
            **({"lora_dropout": lora_dropouts[i]} if lora_dropouts else {}),
        }
        print(
            f"Model {i + 1}: {json.dumps(
                get_updated_fields(
                    SimplePreferenceModel.Config.model_validate(updated_simple_config),
                    SimplePreferenceModel.Config(),
                )
            )}"
        )

    SimplePreferenceModelEnsemble.train_model(updated_config, backing_store)


if __name__ == "__main__":
    ex.observers.append(FileStorageObserver(basedir=backing_store.blob_filepath("logs")))
    ex.run_commandline()
