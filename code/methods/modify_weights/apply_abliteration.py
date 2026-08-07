import os
import time

import torch
from rich.progress import Progress
from torch import Tensor

from code.classes.Config import Config
from code.classes.generators.BaseModel import BaseModel


def apply_model_weight(model_base: BaseModel,
                       refusal_direction: Tensor,
                       abliteration_method,
                       layer: int,
                       get_layer_weight,
                       set_layer_weight):
    layer_tensor = get_layer_weight(layer)
    set_layer_weight(
        weight=torch.nn.Parameter(
            abliteration_method(
                layer_tensor,
                refusal_direction,
                model_base.device
            )
        ).contiguous(),
        layer=layer
    )


def apply_abliteration(config: Config,
                       model_base: BaseModel,
                       refusal_direction: Tensor,
                       method: str,
                       progress: Progress,
                       abliteration_method):
    """
    Apply the abliteration_method to model_base and to the matrices defined in config.
    :param config: The Config with the matrices to modify.
    :param model_base: The model to modify.
    :param refusal_direction: The refusal direction.
    :param method: The abliteration method.
    :param progress: The progressbar to show in the terminal.
    :param abliteration_method: The specific abliteration method to use.
    """

    modified_matrices = get_modified_matrices(config)
    model_dir = f"data/runs/models/{config.model_alias}/{modified_matrices}/{config.model_alias}_abliterated_{method}"
    if os.path.exists(model_dir):
        model_base.load_model(model_dir, set_four_bit_quantization=config.four_bit_quantization)
        return

    if not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))

    start_time = time.time()
    task_modify_tensor = progress.add_task("Modifying tensors...", total=len(model_base.get_layers()))
    for layer in range(len(model_base.get_layers())):
        if config.modify_self_attn_q_proj:
            apply_model_weight(model_base, refusal_direction, abliteration_method, layer,
                               model_base.get_self_attn_q_proj_weight, model_base.set_self_attn_q_proj_weight)

        if config.modify_self_attn_k_proj:
            apply_model_weight(model_base, refusal_direction, abliteration_method, layer,
                               model_base.get_self_attn_k_proj_weight, model_base.set_self_attn_k_proj_weight)

        if config.modify_self_attn_v_proj:
            apply_model_weight(model_base, refusal_direction, abliteration_method, layer,
                               model_base.get_self_attn_v_proj_weight, model_base.set_self_attn_v_proj_weight)

        if config.modify_self_attn_o_proj:
            apply_model_weight(model_base, refusal_direction, abliteration_method, layer,
                               model_base.get_self_attn_o_proj_weight, model_base.set_self_attn_o_proj_weight)

        if config.modify_mlp_down_proj:
            apply_model_weight(model_base, refusal_direction, abliteration_method, layer,
                               model_base.get_mlp_down_proj_weight, model_base.set_mlp_down_proj_weight)

        if config.modify_mlp_gate_proj:
            apply_model_weight(model_base, refusal_direction, abliteration_method, layer,
                               model_base.get_mlp_gate_proj_weight, model_base.set_mlp_gate_proj_weight)

        if config.modify_mlp_up_proj:
            apply_model_weight(model_base, refusal_direction, abliteration_method, layer,
                               model_base.get_mlp_up_proj_weight, model_base.set_mlp_up_proj_weight)

        progress.update(task_modify_tensor, advance=1)

    end_time = time.time()
    progress.update(task_modify_tensor, total=len(model_base.get_layers()))
    print("    Tensor modifying completed in " + str(end_time - start_time) + " seconds.")

    model_base.save_model(model_dir)
    model_base.load_model(model_dir, set_four_bit_quantization=config.four_bit_quantization)


def get_modified_matrices(config: Config):
    s = "Modified_Self_Attn"
    if config.modify_self_attn_q_proj:
        s += "_Q"

    if config.modify_self_attn_k_proj:
        s += "_K"

    if config.modify_self_attn_v_proj:
        s += "_V"

    if config.modify_self_attn_o_proj:
        s += "_O"

    s += "___MLP"

    if config.modify_mlp_down_proj:
        s += "_DOWN"

    if config.modify_mlp_gate_proj:
        s += "_GATE"

    if config.modify_mlp_up_proj:
        s += "_UP"

    return s
