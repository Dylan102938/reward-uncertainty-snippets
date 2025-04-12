# import os
# from typing import Optional

# import torch
# from dotenv import load_dotenv
# from sacred import Experiment
# from transformers import AutoTokenizer

# from reward_uncertainty.datasets import HHRLHFPreprocessor, get_hh_rlhf_dataset
# from reward_uncertainty.observers import FileStorageObserverWithFolders
# from reward_uncertainty.scripts.evaluate_reward_model import (
#     load_peft_model_from_checkpoint,
# )
# from reward_uncertainty.types import DataSubset
# from reward_uncertainty.utils import set_random_seed

# load_dotenv()


# ex = Experiment("train_bayesian_regression_model")
# access_token = os.getenv("HF_ACCESS_TOKEN") or ""


# @ex.config
# def sacred_config():
#     """
#     Path to trained reward model checkpoint.
#     """
#     model_checkpoint: Optional[str] = None  # noqa: F841

#     batch_size: int = 1  # noqa: F841
#     model_name: str = "gpt2"  # noqa: F841

#     """
#     The size of the subset of the training data to use.
#     """
#     train_dataset_size: int = 0  # noqa: F841

#     """
#     The tokneizer for your model, if left empty will use the default.
#     """
#     tokenizer_name: Optional[str] = None  # noqa: F841

#     data_path: str = "Anthropic/hh-rlhf"  # noqa: F841

#     """
#     Which subset of data to use. You can choose
#     between "both", "helpful", and "harmless".
#     """
#     data_subset: DataSubset = "both"  # noqa: F841

#     max_length: int = 1024  # noqa: F841

#     """
#     Whether to use bfloat16 precision.
#     """
#     bf16: bool = True  # noqa: F841

#     seed: Optional[int] = None  # noqa: F841


# @ex.main
# def main(
#     model_checkpoint: str,
#     batch_size: int,
#     model_name: str,
#     train_dataset_size: int,
#     tokenizer_name: Optional[str],
#     data_path: str,
#     data_subset: DataSubset,
#     max_length: int,
#     bf16: bool,
#     seed: Optional[int],
#     **kwargs,
# ):
#     assert (
#         model_checkpoint is not None
#     ), "You must provide a model checkpoint to evaluate."

#     if seed is not None:
#         set_random_seed(seed)

#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=access_token)

#     train_dataset = get_hh_rlhf_dataset(
#         data_subset, "train", data_path, train_dataset_size
#     )

#     model_kwargs = {}
#     if bf16:
#         model_kwargs["dtype"] = torch.bfloat16

#     model = load_peft_model_from_checkpoint(
#         model_checkpoint,
#         model_name,
#         num_outputs=1,
#         **model_kwargs,
#     )

#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     tokenizer.padding_side = "right"
#     model.config.pad_token_id = tokenizer.pad_token_id

#     train_dataset = train_dataset.map(
#         HHRLHFPreprocessor(tokenizer, padding=True, max_length=max_length),
#         batched=True,
#         num_proc=24,
#     )
#     train_dataset = train_dataset.filter()

#     def compute_preds_and_get_weights(example):
#         output = {}
#         for key in ["chosen", "rejected"]:
#             batch = tokenizer.pad(
#                 {
#                     "input_ids": example[f"input_ids_{key}"],
#                 },
#                 padding=True,
#                 max_length=max_length,
#                 pad_to_multiple_of=64,
#                 return_tensors="pt",
#             )
#             last_token_indices = batch["attention_mask"].sum(dim=1) - 1
#             last_token_idx = last_token_indices[0].item()
#             with torch.no_grad():
#                 outputs = model(
#                     input_ids=batch["input_ids"].to("cuda"),
#                     attention_mask=batch["attention_mask"].to("cuda"),
#                     output_hidden_states=True,
#                     return_dict=True,
#                 )
#                 hidden_states = outputs.hidden_states
#                 last_hidden_layer = hidden_states[last_token_idx]

#                 print("Last hidden layer shape:", last_hidden_layer.shape)

#             output[f"reward_output_{key}"] = outputs
#             output[f"hidden_states_{key}"] = last_hidden_layer.tolist()

#         return output

#     train_results = train_dataset.map(
#         compute_preds_and_get_weights,
#         remove_columns=[
#             "input_ids_chosen",
#             "input_ids_rejected",
#             "attention_mask_chosen",
#             "attention_mask_rejected",
#         ],
#         batched=True,
#         batch_size=batch_size,
#     )

#     def expand_hidden_states(example):
#         hs_list = example["hidden_states_chosen"]
#         for subset in ["chosen", "rejected"]:
#             for i, state in enumerate(hs_list):
#                 example[f"hs_{subset}_{i}"] = state

#         return example

#     train_results = train_results.map(
#         expand_hidden_states,
#         remove_columns=["hidden_states_chosen", "hidden_states_rejected"],
#     )

#     train_results.to_csv("train_results.csv")


# if __name__ == "__main__":
#     ex.observers.append(FileStorageObserverWithFolders("results/evals"))
#     ex.run_commandline()
