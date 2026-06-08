import contextlib
import functools
from typing import List, Tuple, Callable

import torch
from jaxtyping import Float
from torch import Tensor


@contextlib.contextmanager
def add_hooks(
        module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
        module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
        **kwargs
):
    """
    Temporarily registers forward hooks and forward pre-hooks
    for specified PyTorch modules.

    The context manager automatically installs hooks before
    entering the context and removes them afterward, even if
    an exception occurs.

    This mechanism allows temporary interception of model
    activations without permanently modifying the model.

    Parameters
    ----------
    module_forward_pre_hooks : List[Tuple[Module, Callable]]
        List containing module and pre-hook pairs.

    module_forward_hooks : List[Tuple[Module, Callable]]
        List containing module and forward-hook pairs.

    **kwargs
        Additional keyword arguments passed to hook functions.

    Yields
    ------
    None
        Executes enclosed code block with hooks enabled.
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn


def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn
