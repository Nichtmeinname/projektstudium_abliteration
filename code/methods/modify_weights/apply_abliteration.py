import os

import torch

from code.classes.Config import Config
from code.classes.generators.BaseModel import BaseModel
from code.methods.modify_weights.modify_tensors import modify_tensor_norm_preserved


def apply_abliteration(config: Config, model_base: BaseModel, refusal_direction: torch.Tensor):
    state_dict_file = f"../../data/runs/state_dicts/{config.model_alias}/abliteration_state_dict.pth"
    if os.path.exists(state_dict_file):
        return torch.load(state_dict_file)

    if not os.path.exists(os.path.dirname(state_dict_file)):
        os.makedirs(os.path.dirname(state_dict_file))

    for layer in range(len(model_base.get_layers())):
        layer_attn_w_q_proj = model_base.get_layers()[layer].self_attn.q_proj.weight
        model_base.model.model.layers[layer].self_attn.q_proj.weight = torch.nn.Parameter(
            modify_tensor_norm_preserved(layer_attn_w_q_proj,
                                         refusal_direction,
                                         model_base.device)
        ).contiguous()

        layer_mlp_down_w_q_proj = model_base.get_layers()[layer].mlp.down_proj.weight
        model_base.model.model.layers[layer].mlp.down_proj.weight = torch.nn.Parameter(
            modify_tensor_norm_preserved(layer_mlp_down_w_q_proj,
                                         refusal_direction,
                                         model_base.device)
        ).contiguous()

    state_dict = model_base.model.state_dict()
    torch.save(state_dict, state_dict_file)

    return state_dict
