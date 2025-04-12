import json
import os
from typing import Optional

import torch
from datasets import Dataset
from dotenv import load_dotenv
from sacred import Experiment
from sacred.observers import FileStorageObserver

from experiments.utils.data_processing.datasets import get_hh_rlhf_dataset
from experiments.utils.data_processing.preprocessors.hh_rlhf import HHRLHFPreprocessor
from experiments.utils.misc import get_tokenizer, get_updated_fields
from reward_uncertainty.model.simple_preference_model import SimplePreferenceModel
from reward_uncertainty.store.disk_store import DiskBackingStore
from reward_uncertainty.types import DataSubset
from reward_uncertainty.utils import (
    bootstrapped_dataset,
    set_random_seed,
    shuffled_dataset,
)

load_dotenv()
ex = Experiment("train_single_preference_model")
access_token = os.getenv("HF_ACCESS_TOKEN") or ""
backing_store = DiskBackingStore.create(SimplePreferenceModel.generate_id())


@ex.config
def sacred_config():
    """
    Used for multi-gpu
    """
    local_rank: int = -1  # noqa: F841

    """
    If you want to resume training where it left off.
    """
    resume_from_checkpoint: bool = False  # noqa: F841

    """
    Path to deepspeed config if using deepspeed. You
    may need this if the model that you want to train
    doesn't fit on a single GPU.
    """
    deepspeed: Optional[str] = None  # noqa: F841

    per_device_train_batch_size: int = 8  # noqa: F841
    per_device_eval_batch_size: int = 1  # noqa: F841
    gradient_accumulation_steps: int = 1  # noqa: F841
    learning_rate: float = 3e-6  # noqa: F841
    weight_decay: float = 0.001  # noqa: F841

    """
    The model that you want to train from the Hugging
    Face hub. E.g. gpt2, gpt2-xl, bert, etc.
    """
    model_name: str = "gpt2"  # noqa: F841

    data_path: str = "Anthropic/hh-rlhf"  # noqa: F841

    """
    Which subset of data to use. You can choose
    between "both", "helpful", and "harmless".
    """
    data_subset: DataSubset = "both"  # noqa: F841

    """
    The number of atoms to use for the categorical
    reward model.
    """
    num_atoms: int = 10  # noqa: F841

    """
    The entropy coefficient for the categorical reward
    model.
    """
    entropy_coeff: float = 0.1  # noqa: F841

    """
    The variance penalty for the mean and variance
    reward model.
    """
    variance_penalty: float = 0.0  # noqa: F841

    """
    The tokenizer for your model, if left empty will
    use the default.
    """
    tokenizer_name: Optional[str] = None  # noqa: F841

    bf16: bool = True  # noqa: F841

    """
    The number of training epochs for the reward
    model.
    """
    num_train_epochs: int = 1  # noqa: F841

    """
    The size of the subset of the training data to use.
    """
    train_dataset_size: int = 0  # noqa: F841

    """
    Whether or not the train_dataset should be shuffled.
    """
    shuffle_train_dataset: bool = True  # noqa: F841

    """
    The size of the subset of the eval data to use.
    """
    eval_dataset_size: int = 0  # noqa: F841

    """
    Enables gradient checkpointing.
    """
    gradient_checkpointing: bool = False  # noqa: F841

    """
    The optimizer to use.
    """
    optim: str = "adamw_hf"  # noqa: F841

    """
    The lr scheduler.
    """
    lr_scheduler_type: str = "cosine"  # noqa: F841

    max_length: int = 1024  # noqa: F841

    """
    Whether to run eval after the first step.
    """
    eval_first_step: bool = False  # noqa: F841

    """
    LORA hyperparameters
    """
    lora_rank: int = 64  # noqa: F841
    lora_alpha: int = 128  # noqa: F841
    lora_dropout: float = 0.1  # noqa: F841

    """
    Whether or not to bootstrap the training dataset.
    """
    use_bootstrapped_dataset: bool = False  # noqa: F841

    seed: Optional[int] = None  # noqa: F841


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
    data_subset: DataSubset,
    data_path: str,
    train_dataset_size: int,
    shuffle_train_dataset: bool,
    eval_dataset_size: int,
    model_name: str,
    learning_rate: float,
    lr_scheduler_type: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    weight_decay: float,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    deepspeed: Optional[str],
    local_rank: int,
    bf16: bool,
    optim: str,
    tokenizer_name: Optional[str],
    max_length: int,
    resume_from_checkpoint: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_bootstrapped_dataset: bool,
    seed: Optional[int],
    entropy_coeff: float,
):
    assert ex.current_run is not None
    print("Starting Experiment: train_single_preference_model")

    torch.set_anomaly_enabled(True)

    if seed is not None:
        set_random_seed(seed)

    tokenizer_name = tokenizer_name if tokenizer_name else model_name

    train_dataset = get_hh_rlhf_dataset(data_subset, "train", data_path, train_dataset_size)
    train_dataset = shuffled_dataset(train_dataset) if shuffle_train_dataset else train_dataset
    train_dataset = bootstrapped_dataset(train_dataset) if use_bootstrapped_dataset else train_dataset
    train_dataset = preprocess_dataset(train_dataset, tokenizer_name, max_length)

    eval_dataset = get_hh_rlhf_dataset(data_subset, "test", data_path, eval_dataset_size)
    eval_dataset = preprocess_dataset(eval_dataset, tokenizer_name, max_length)

    updated_config = SimplePreferenceModel.TrainingConfig(
        access_token=access_token,
        train_dataset_name=f"hh_rlhf_{data_subset}",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        weight_decay=weight_decay,
        model_name=model_name,
        entropy_coeff=entropy_coeff,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        deepspeed=deepspeed,
        local_rank=local_rank,
        bf16=bf16,
        optim=optim,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        resume_from_checkpoint=resume_from_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    print("====== Updated Preference Model Config ======")
    print(
        json.dumps(
            get_updated_fields(
                SimplePreferenceModel.Config.model_validate(updated_config.model_dump()),
                SimplePreferenceModel.Config(),
            ),
            indent=2,
        )
    )

    SimplePreferenceModel.train_model(updated_config, backing_store)


if __name__ == "__main__":
    ex.observers.append(FileStorageObserver(basedir=backing_store.blob_filepath("logs")))
    ex.run_commandline()
