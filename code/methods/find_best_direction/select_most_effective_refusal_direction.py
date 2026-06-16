import functools
import json
import math
import os
from itertools import product

import torch
from jaxtyping import Float, Int
from torch import Tensor

from code.classes.Config import Config
from code.classes.generators.BaseModel import BaseModel
from code.utils.hookutils import get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook, add_hooks


def refusal_score(
        logits: Float[Tensor, 'batch seq d_vocab_out'],
        refusal_toks: Int[Tensor, 'batch seq'],
        epsilon: Float = 1e-8,
):
    logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)

    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)


def get_refusal_scores(config: Config, model_base: BaseModel, prompts: list, fwd_pre_hooks=[],
                       fwd_hooks=[]):
    refusal_toks = model_base.get_refusal_toks()
    refusal_score_fn = functools.partial(refusal_score, refusal_toks=refusal_toks)

    refusal_scores = torch.zeros(len(prompts), device=model_base.device)

    for i in range(0, len(prompts), config.batch_size):
        tokenized_instructions = model_base.tokenize_prompt(prompts=prompts[i:i + config.batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model_base.model(
                input_ids=tokenized_instructions.input_ids.to(model_base.device),
                attention_mask=tokenized_instructions.attention_mask.to(model_base.device),
            ).logits

        refusal_scores[i:i + config.batch_size] = refusal_score_fn(logits=logits)

    return refusal_scores


def evaluate_pos_layer(config: Config, model_base: BaseModel, direction: Float[Tensor, "d_model"], val_prompts: list):
    fwd_pre_hooks = [
        (model_base.get_layers()[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer
        in range(model_base.get_num_of_hidden_layers())]
    fwd_hooks = [(model_base.get_attn_modules()[layer], get_direction_ablation_output_hook(direction=direction)) for
                 layer in range(model_base.get_num_of_hidden_layers())]
    fwd_hooks += [(model_base.get_mlp_modules()[layer], get_direction_ablation_output_hook(direction=direction)) for
                  layer in range(model_base.get_num_of_hidden_layers())]

    refusal_scores = get_refusal_scores(config, model_base, val_prompts, fwd_pre_hooks=fwd_pre_hooks,
                                        fwd_hooks=fwd_hooks)

    # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
    return -refusal_scores.mean().item()


def select_direction(config: Config, model_base: BaseModel, harmful_val: list, harmless_val: list,
                     mean_diffs: Float[Tensor, "n_positions n_layers d_model"]):
    n_pos, n_layer, _ = mean_diffs.shape
    layer_scores = {}
    for pos_idx, layer_idx in product(range(-n_pos, 0), range(n_layer)):
        print(f"    Testing position {pos_idx} and layer {layer_idx}.")
        abliteration_direction = mean_diffs[pos_idx, layer_idx]
        rate = evaluate_pos_layer(
            config, model_base, abliteration_direction, harmful_val + harmless_val
        )
        if not math.isnan(rate):
            layer_scores[pos_idx, layer_idx] = rate

    # Best layer = lowest refusal after abliteration
    best_layer = max(layer_scores, key=layer_scores.get)
    print(f"    Bester Pos/Layer: {best_layer} (Ablehnungsrate: {layer_scores[best_layer]:.2f})")

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
        with open(json_file_name, "r") as f:
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
