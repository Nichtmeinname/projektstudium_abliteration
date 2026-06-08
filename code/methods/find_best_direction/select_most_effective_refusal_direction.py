import gc
import json
import os
from itertools import product

import torch
from jaxtyping import Float
from torch import Tensor

from code.classes.Config import Config
from code.classes.RefusalDetector import RefusalDetector
from code.classes.generators.BaseModel import BaseModel


def evaluate_pos_layer(config: Config, model_base: BaseModel, layer_idx: int,
                       direction: Float[Tensor, "d_model"], val_prompts: list, detector: RefusalDetector):
    # Temporären Hook nur für diesen einen Layer registrieren
    def ablation_hook(module, input, output):
        def ablation_hook(module, input, output):

            # Tensor oder Tupel behandeln
            if isinstance(output, tuple):
                residual = output[0]
            else:
                residual = output

            d = direction.to(residual.device)

            was_2d = False

            if residual.dim() == 2:
                residual = residual.unsqueeze(0)
                was_2d = True

            projection = (residual * d).sum(dim=-1)

            residual_new = (
                    residual
                    - projection.unsqueeze(-1) * d
            )

            if was_2d:
                residual_new = residual_new.squeeze(0)

            # Ursprüngliches Format zurückgeben
            if isinstance(output, tuple):
                return (residual_new,) + output[1:]

            return residual_new

    handle = model_base.register_forward_hook_on_layer(ablation_hook, layer_idx)

    refusals = 0

    responses = model_base.generate_multiple(val_prompts, batch_size=config.batch_size)

    evaluated_responses = detector.detect(responses)

    for evaluated_response in evaluated_responses:
        if evaluated_response["response_type"] != "No Refusal":
            refusals += 1

    handle.remove()

    refusal_rate = refusals / len(val_prompts)
    return refusal_rate


def select_direction(config: Config, model_base: BaseModel, harmful_val: list, harmless_val: list,
                     mean_diffs: Float[Tensor, "n_positions n_layers d_model"]):
    n_pos, n_layer, _ = mean_diffs.shape
    layer_scores = {}
    detector = RefusalDetector()
    for pos_idx, layer_idx in product(range(-n_pos, 0), range(n_layer)):
        print(f"Testing position {pos_idx} and layer {layer_idx}.")
        abliteration_direction = mean_diffs[pos_idx, layer_idx]
        rate = evaluate_pos_layer(
            config, model_base, layer_idx, abliteration_direction, harmful_val + harmless_val, detector
        )
        layer_scores[pos_idx, layer_idx] = rate

    # Bester Layer = niedrigste Ablehnungsrate nach Abliteration
    best_layer = min(layer_scores, key=layer_scores.get)
    print(f"    Bester Pos/Layer: {best_layer} (Ablehnungsrate: {layer_scores[best_layer]:.2f})")

    del detector
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return best_layer[0], best_layer[1], mean_diffs[best_layer]


def select_most_effective_refusal_direction(config: Config, model_base: BaseModel,
                                            harmful_val: list, harmless_val: list,
                                            mean_diffs: Float[Tensor, "n_positions n_layers d_model"]):
    direction_dir = "../../data/runs/selected_refusal_direction/" + config.model_alias
    json_file_name = os.path.join(direction_dir, "direction_metadata.json")
    direction_file_name = os.path.join(direction_dir, "direction.pt")
    if not os.path.exists(direction_dir):
        os.makedirs(direction_dir)

    if os.path.exists(direction_file_name) and os.path.exists(json_file_name):
        with open(json_file_name, "w") as f:
            pos_and_layer = json.load(f)
        return pos_and_layer["pos"], pos_and_layer["layer"], torch.load(direction_file_name)

    pos, layer, direction = select_direction(
        config,
        model_base,
        harmful_val,
        harmless_val,
        mean_diffs
    )

    with open(json_file_name, "w") as f:
        json.dump({"pos": pos, "layer": layer}, f)

    torch.save(direction, direction_file_name)

    return pos, layer, direction
