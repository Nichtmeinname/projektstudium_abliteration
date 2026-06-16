import os
from typing import List

import torch
from jaxtyping import Float
from torch import Tensor

from code.classes.Config import Config
from code.classes.generators.BaseModel import BaseModel
from code.utils.hookutils import add_hooks


def get_mean_activations_pre_hook(layer: int, cache: Float[Tensor, "pos layer d_model"], n_samples: int,
                                  positions: List[int]):
    """
    Creates a forward pre-hook for collecting mean activations of a
    specific transformer layer.

    The returned hook function is executed immediately before the
    forward pass of a layer. It extracts the layer input activations
    for selected token positions and incrementally computes their
    mean value across multiple samples.

    Instead of storing activations for every prompt individually,
    the mean is updated directly using an online averaging approach,
    which reduces memory usage.

    Parameters
    ----------
    layer : int
        Index of the transformer layer currently being processed.

    cache : Tensor
        Tensor used to store the accumulated mean activations.
        Shape:
        (n_positions, n_layers, d_model)

    n_samples : int
        Total number of prompts used for averaging.

    positions : List[int]
        Token positions whose activations should be extracted.

    Returns
    -------
    Callable
        Forward pre-hook function for a transformer layer.
    """

    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)

    return hook_fn


def get_mean_activations(config: Config, model: BaseModel, prompts: list):
    """
    Computes mean activations across multiple prompts for all
    transformer layers of a language model.

    For each prompt, a forward pass through the model is executed
    while forward pre-hooks collect activations from selected token
    positions. Activations are accumulated and averaged over all
    samples.

    The resulting tensor stores mean activation vectors for each
    layer and token position.

    Parameters
    ----------
    config: The config.

    model : QwenLLMModel
        Generator object containing the model, tokenizer,
        and helper functions.

    prompts : list[str]
        List of prompts used to compute mean activations.

    Returns
    -------
    Tensor
        Mean activations tensor with shape:

        (n_positions, n_layers, d_model)

    where:

    - n_positions = number of selected token positions
    - n_layers = number of transformer layers
    - d_model = hidden size of the model
    """
    torch.cuda.empty_cache()

    positions = list(range(-len(model.get_eoi_toks()), 0))
    n_positions = len(positions)
    n_layers = model.get_num_of_hidden_layers()
    n_samples = len(prompts)
    d_model = model.get_hidden_size()

    # we store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    block_modules = model.get_layers()
    fwd_pre_hooks = [(block_modules[layer],
                      get_mean_activations_pre_hook(layer=layer, cache=mean_activations, n_samples=n_samples,
                                                    positions=positions)) for layer in range(n_layers)]

    for i in range(0, len(prompts), config.batch_size):
        input = model.tokenize_prompts(prompts[i:i + config.batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model.model(
                input_ids=input.input_ids.to(model.device),
                attention_mask=input.attention_mask.to(model.device),
            )

    return mean_activations


def get_mean_diff(config: Config, model: BaseModel, harmful_train: list, harmless_train: list):
    mean_activations_harmful = get_mean_activations(config, model, harmful_train)
    mean_activations_harmless = get_mean_activations(config, model, harmless_train)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless

    return mean_diff


def generate_direction(config: Config, model: BaseModel, harmful_train: list, harmless_train: list, path_to_mean_diff_dir: str):
    """
    Generate mean diffs between harmful and harmless.
    :param config: The config.
    :param model: The BaseModel with the model to calculate mean diffs.
    :param harmful_train: The harmful train set.
    :param harmless_train: The harmless train set.
    :param path_to_mean_diff_dir: The path to save mean diffs.
    :return: The mean diffs between harmful and harmless of the BaseModel model.
    """
    if not os.path.exists(path_to_mean_diff_dir):
        os.makedirs(path_to_mean_diff_dir)

    mean_diffs = get_mean_diff(config, model, harmful_train, harmless_train)

    assert mean_diffs.shape == (len(model.get_eoi_toks()), model.model.config.num_hidden_layers,
                                model.model.config.hidden_size)
    assert not mean_diffs.isnan().any()

    return mean_diffs
