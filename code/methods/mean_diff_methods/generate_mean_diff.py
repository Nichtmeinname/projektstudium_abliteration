import os

import torch

from code.classes.Config import Config
from code.classes.generators.BaseModel import BaseModel
from code.methods.mean_diff_methods.generate_direction import generate_direction


def generate_mean_diff(config: Config, model: BaseModel, harmful_train: list, harmless_train: list):
    """
    Calculates the mean difference between the harmful train and harmless train.

    :param config: The configuration of the model.
    :param model: The trained model.
    :param harmful_train: The harmful train.
    :param harmless_train: The harmless train.
    :return: The mean differences between the harmful train and harmless train.
    """
    directions_dir = "../../data/runs/generated_directions"
    if not os.path.exists(directions_dir):
        os.makedirs(directions_dir)

    path_to_saved_mean_diffs = os.path.join(directions_dir, config.model_alias, "mean_diffs.pt")
    if os.path.exists(path_to_saved_mean_diffs):
        return torch.load(path_to_saved_mean_diffs)

    mean_diffs = generate_direction(model, harmful_train, harmless_train, os.path.dirname(path_to_saved_mean_diffs))

    torch.save(mean_diffs, path_to_saved_mean_diffs)

    return mean_diffs
