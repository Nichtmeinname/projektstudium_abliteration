import os
import time

import torch

from code.classes.Config import Config
from code.classes.generators.BaseModel import BaseModel
from code.methods.modify_weights.modify_tensors import modify_tensor_norm_preserved, modify_tensor_standard


def apply_abliteration_norm_preserving(config: Config, model_base: BaseModel, refusal_direction: torch.Tensor,
                                       method: str):
    model_dir = f"../../data/runs/models/{config.model_alias}/{config.model_alias}_abliterated_{method}"
    if os.path.exists(model_dir):
        model_base.load_model(model_dir, set_four_bit_quantization=config.four_bit_quantization)
        return

    if not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))

    start_time = time.time()
    for layer in range(len(model_base.get_layers())):
        layer_attn_w_q_proj = model_base.get_attn_q_proj_weight(layer)
        model_base.set_attn_q_proj_weight(
            weight=torch.nn.Parameter(
                modify_tensor_norm_preserved(
                    layer_attn_w_q_proj,
                    refusal_direction,
                    model_base.device
                )
            ).contiguous(),
            layer=layer
        )

        layer_mlp_down_w_q_proj = model_base.get_mlp_down_proj_weight(layer)
        model_base.set_mlp_down_proj_weight(
            weight=torch.nn.Parameter(
                modify_tensor_norm_preserved(
                    layer_mlp_down_w_q_proj,
                    refusal_direction,
                    model_base.device
                )
            ).contiguous(),
            layer=layer
        )

    end_time = time.time()
    print("    Abliteration (tensor modifying) completed in " + str(end_time - start_time) + " seconds.")

    model_base.save_model(model_dir)
    model_base.load_model(model_dir, set_four_bit_quantization=config.four_bit_quantization)


def apply_abliteration_standard(config: Config, model_base: BaseModel, refusal_direction: torch.Tensor, method: str):
    model_dir = f"../../data/runs/models/{config.model_alias}/{config.model_alias}_abliterated_{method}"
    if os.path.exists(model_dir):
        model_base.load_model(model_dir, set_four_bit_quantization=config.four_bit_quantization)
        return

    if not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))

    start_time = time.time()
    for layer in range(len(model_base.get_layers())):
        layer_attn_w_q_proj = model_base.get_attn_q_proj_weight(layer)
        model_base.set_attn_q_proj_weight(
            weight=torch.nn.Parameter(
                modify_tensor_standard(
                    layer_attn_w_q_proj,
                    refusal_direction,
                    model_base.device
                )
            ).contiguous(),
            layer=layer
        )

        layer_mlp_down_w_q_proj = model_base.get_mlp_down_proj_weight(layer)
        model_base.set_mlp_down_proj_weight(
            weight=torch.nn.Parameter(
                modify_tensor_standard(
                    layer_mlp_down_w_q_proj,
                    refusal_direction,
                    model_base.device
                )
            ).contiguous(),
            layer=layer
        )

    end_time = time.time()
    print("    Abliteration (tensor modifying) completed in " + str(end_time - start_time) + " seconds.")

    model_base.save_model(model_dir)
    model_base.load_model(model_dir, set_four_bit_quantization=config.four_bit_quantization)
