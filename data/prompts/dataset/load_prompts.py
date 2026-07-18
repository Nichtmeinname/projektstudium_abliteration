import json
import os
import random


def load_prompts(n_samples: int, harm_type: str, dataset_type: str = "test", seed: int = 42,
                 instructions_only: bool = False):
    """
    Loads prompts from a JSON file.

    :param n_samples: The number of prompts to load.
    :param harm_type: The type of harmonization to use (harmful, harmless).
    :param dataset_type: The type of dataset to use (train, val, test).
    :param seed: The random seed to use.
    :param instructions_only: The instructions only. Returns a list of prompts, otherwise JSON.
    :return: If instructions_only is True, returns a list of prompts. Otherwise, returns a `JSON of prompts.
    """
    dataset_dir_path = os.path.dirname(os.path.realpath(__file__))
    json_file_path = os.path.join(dataset_dir_path, f"splits/{harm_type}_{dataset_type}.json")
    with open(json_file_path, "r") as f:
        prompts = json.load(f)

    random.seed(seed)

    prompts = random.sample(prompts, n_samples)

    if instructions_only:
        prompts = [prompt["instruction"] for prompt in prompts]
    return prompts
