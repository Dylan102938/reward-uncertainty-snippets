import os
from typing import Optional

from datasets import Dataset
from dotenv import load_dotenv
from sacred import Experiment
from sacred.observers import FileStorageObserver

from experiments.utils.data_processing.datasets import get_eval_dataset
from experiments.utils.misc import get_tokenizer
from reward_uncertainty.eval.reward_eval import RewardEval
from reward_uncertainty.model.simple_preference_model import SimplePreferenceModel
from reward_uncertainty.store.disk_store import DiskBackingStore

load_dotenv()
ex = Experiment("evaluate_preference_model")
access_token = os.getenv("HF_ACCESS_TOKEN") or ""
backing_store = DiskBackingStore.create(RewardEval.generate_id())


@ex.config
def sacred_config():
    reward_model_id: Optional[str] = None  # noqa: F841
    dataset: Optional[str] = None  # noqa: F841
    max_length: int = 1024  # noqa: F841
    batch_size: int = 1  # noqa: F841
    model_checkpoint: Optional[str] = None
    use_latest_checkpoint: bool = model_checkpoint is None  # noqa: F841


@ex.main
def main(
    reward_model_id: Optional[str],
    dataset: Optional[str],
    max_length: int,
    batch_size: int,
    model_checkpoint: Optional[str],
    use_latest_checkpoint: bool,
):
    assert reward_model_id is not None, "reward_model_id is required"
    assert dataset is not None, "dataset is required"
    assert ex.current_run is not None, "Run should be defined at this point."

    store = DiskBackingStore.get(reward_model_id)
    if store is None:
        raise ValueError(
            "Please make sure you select a valid reward model with a BackingStore of type PossibleDiskBackingStore."
        )

    reward_model = SimplePreferenceModel.from_backing_store(store)

    tokenizer_name = reward_model.config.tokenizer_name or reward_model.config.model_name
    tokenizer = get_tokenizer(tokenizer_name, access_token)

    eval_dataset: Dataset = get_eval_dataset(dataset, tokenizer=tokenizer, padding=True, max_length=max_length)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
    )

    RewardEval.create_with_model(
        reward_model,
        RewardEval.EvalConfig(
            eval_dataset=eval_dataset,
            dataset_name=dataset,
            prediction_config=SimplePreferenceModel.PredictionConfig(
                access_token=access_token,
                model_checkpoint=model_checkpoint,
                use_latest_checkpoint=use_latest_checkpoint,
                max_length=max_length,
                batch_size=batch_size,
                num_outputs=1,
            ),
        ),
        backing_store,
    )


if __name__ == "__main__":
    ex.observers.append(FileStorageObserver(basedir=backing_store.blob_filepath("logs")))
    ex.run_commandline()
