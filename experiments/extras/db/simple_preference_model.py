from typing import cast

from peewee import FloatField, ForeignKeyField, IntegerField, TextField

from experiments.extras.db.common import SharedModel
from experiments.extras.db.simple_preference_model_ensemble import (
    SimplePreferenceModelEnsembleArtifact,
)


class SimplePreferenceModelArtifact(SharedModel):
    model_id: TextField
    train_dataset = TextField(null=False)
    per_device_train_batch_size = IntegerField(null=False)
    gradient_accumulation_steps = IntegerField(null=False)
    learning_rate = FloatField(null=False)
    weight_decay = FloatField(null=False)
    model_name = TextField(null=False)
    entropy_coeff = FloatField(null=False)
    tokenizer_name = TextField(null=True)
    num_train_epochs = IntegerField(null=False)
    lr_scheduler_type = TextField(null=False)
    lora_rank = IntegerField(null=False)
    lora_alpha = FloatField(null=False)
    lora_dropout = FloatField(null=False)
    max_length = IntegerField(null=False)
    _ensemble = ForeignKeyField(
        SimplePreferenceModelEnsembleArtifact,
        null=True,
        backref="_members",
        on_delete="CASCADE",
        object_id_name="ensemble_id",
    )

    class Meta:
        table_name = "simple_preference_model"

    @property
    def ensemble(self) -> SimplePreferenceModelEnsembleArtifact:
        return cast(SimplePreferenceModelEnsembleArtifact, self._ensemble)
