import os
import shutil

from pydantic import BaseModel
from sacred.observers import FileStorageObserver
from transformers import AutoTokenizer
from typing_extensions import TypeVar

from reward_uncertainty.store.base_store import BackingStore


def get_tokenizer(tokenizer_name: str, access_token: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    return tokenizer


def copy_observer_files_to_backing_store(
    observer: FileStorageObserver, backing_store: BackingStore, delete_observer_dir: bool = False
):
    assert observer.dir is not None

    for root, _, files in os.walk(observer.dir):
        for file in files:
            content = open(os.path.join(root, file), "rb").read()
            blob_id = os.path.join("observer", root, file)
            backing_store.create_blob(blob_id, content)

            print(f"Copied {os.path.join(root, file)} to {blob_id}")

    if delete_observer_dir:
        shutil.rmtree(observer.dir)
        print(f"Deleted observer directory {observer.dir}")


T = TypeVar("T", bound=BaseModel)


def get_updated_fields(updated_config: T, base_config: T) -> dict:
    updated_dict = updated_config.model_dump()
    base_dict = base_config.model_dump()

    return {k: v for k, v in updated_dict.items() if v != base_dict[k]}
