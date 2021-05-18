

import numpy as np
import torch
import transformers
from transformers.trainer_callback import TrainerCallback, TrainingArguments, TrainerControl, TrainerState

class DisableCheckpointCallbackHandler(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save=False
        return control