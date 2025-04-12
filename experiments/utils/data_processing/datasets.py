from typing import Literal

from datasets import Dataset, concatenate_datasets, load_dataset

from experiments.utils.data_processing.preprocessors.hh_rlhf import HHRLHFPreprocessor
from experiments.utils.data_processing.preprocessors.hh_rlhf_jailbroken import (
    HHRLHFJailBrokenPreprocessor,
)
from experiments.utils.data_processing.preprocessors.safe_rlhf import (
    SafeRLHFPreprocessor,
)
from experiments.utils.data_processing.preprocessors.strongreject_jailbroken import (
    SRJailbrokenPreprocessor,
)
from reward_uncertainty.types import DataSubset, SafeRLHFDataSubset

HH_RLHF_JAILBREAK_DATASET_PATH = "/nas/ucb/cassidy/inconsistent/data/jailbroken_responses.jsonl"
STRONGREJECT_JAILBREAK_DATASET_PATH = "/nas/ucb/dfeng/data/strongreject_jailbroken.jsonl"


def get_hh_rlhf_dataset(
    data_subset: DataSubset,
    split: Literal["train", "test"],
    data_path="Anthropic/hh-rlhf",
    dataset_size: int = 0,
) -> Dataset:
    datasets: list[Dataset] = []
    if data_path == "Anthropic/hh-rlhf":
        if data_subset == "harmless" or data_subset == "both":
            datasets.append(
                load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split).map(
                    lambda data: {"data_subset": "harmless"}
                )
            )
        if data_subset == "helpful" or data_subset == "both":
            datasets.append(
                load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split).map(
                    lambda data: {"data_subset": "helpful"}
                )
            )
    else:
        datasets.append(load_dataset(data_path, split=split).map(lambda data: {"data_subset": "both"}))

    if dataset_size > 0:
        datasets = [d.select(range(dataset_size // len(datasets))) for d in datasets]

    return concatenate_datasets(datasets)


def get_safe_rlhf_dataset(split: Literal["train", "test"], data_subset: SafeRLHFDataSubset = "default") -> Dataset:
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split, name=data_subset)

    true_false_pairs = dataset.filter(
        lambda row: row["is_response_0_safe"] is True and row["is_response_1_safe"] is False
    )
    false_true_pairs = dataset.filter(
        lambda row: row["is_response_0_safe"] is False and row["is_response_1_safe"] is True
    )

    true_false_pairs = true_false_pairs.map(
        lambda examples: {
            "prompt": examples["prompt"],
            "chosen_response": examples["response_0"],
            "rejected_response": examples["response_1"],
            "prompt_source": examples["prompt_source"],
            "chosen_source": examples["response_0_source"],
            "rejected_source": examples["response_1_source"],
            "harm_category": examples["response_1_harm_category"],
            "severity_level": examples["response_1_severity_level"],
        },
        remove_columns=true_false_pairs.column_names,
    )
    false_true_pairs = false_true_pairs.map(
        lambda examples: {
            "prompt": examples["prompt"],
            "chosen_response": examples["response_1"],
            "rejected_response": examples["response_0"],
            "prompt_source": examples["prompt_source"],
            "chosen_source": examples["response_1_source"],
            "rejected_source": examples["response_0_source"],
            "harm_category": examples["response_0_harm_category"],
            "severity_level": examples["response_0_severity_level"],
        },
        remove_columns=false_true_pairs.column_names,
    )

    return concatenate_datasets([true_false_pairs, false_true_pairs], axis=0)


def get_eval_dataset(dataset_name: str, **preprocessor_kwargs) -> Dataset:
    if dataset_name == "hh_rlhf_both":
        eval_dataset = get_hh_rlhf_dataset(
            "both",
            split="test",
            data_path="Anthropic/hh-rlhf",
        )
        preprocessor = HHRLHFPreprocessor(**preprocessor_kwargs)
    elif dataset_name == "hh_rlhf_helpful":
        eval_dataset = get_hh_rlhf_dataset(
            "helpful",
            split="test",
            data_path="Anthropic/hh-rlhf",
        )
        preprocessor = HHRLHFPreprocessor(**preprocessor_kwargs)
    elif dataset_name == "hh_rlhf_harmless":
        eval_dataset = get_hh_rlhf_dataset("harmless", split="test", data_path="Anthropic/hh-rlhf")
        preprocessor = HHRLHFPreprocessor(**preprocessor_kwargs)
    elif dataset_name == "hh_rlhf_jailbroken":
        eval_dataset = Dataset.from_json(HH_RLHF_JAILBREAK_DATASET_PATH)
        preprocessor = HHRLHFJailBrokenPreprocessor(**preprocessor_kwargs)
    elif dataset_name == "strongreject_jailbroken":
        eval_dataset = Dataset.from_json(STRONGREJECT_JAILBREAK_DATASET_PATH)
        preprocessor = SRJailbrokenPreprocessor(**preprocessor_kwargs)
    elif dataset_name == "safe_rlhf":
        eval_dataset = get_safe_rlhf_dataset(split="test")
        preprocessor = SafeRLHFPreprocessor(**preprocessor_kwargs)
    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name")

    eval_dataset = eval_dataset.map(preprocessor.run, batched=True, num_proc=24)
    eval_dataset = eval_dataset.select_columns(
        [
            "chosen",
            "rejected",
            "input_ids_chosen",
            "input_ids_rejected",
            "attention_mask_chosen",
            "attention_mask_rejected",
        ]
    )
    eval_dataset = eval_dataset.filter()

    return eval_dataset
