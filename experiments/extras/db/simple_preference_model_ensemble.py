from typing import TYPE_CHECKING

from peewee import BooleanField, IntegerField, ModelSelect, TextField

from experiments.extras.db.common import SharedModel

if TYPE_CHECKING:
    from experiments.extras.db.simple_preference_model import (
        SimplePreferenceModelArtifact,
    )


class SimplePreferenceModelEnsembleArtifact(SharedModel):
    model_id: TextField
    train_dataset = TextField(null=False)
    num_ensemble_members = IntegerField(null=False)
    shuffle_train_dataset = BooleanField(null=False)
    bootstrap_train_dataset = BooleanField(null=False)
    _members: ModelSelect

    @property
    def members(self) -> list[SimplePreferenceModelArtifact]:
        return list(self._members)
