import os

import torch
from huggingface_hub import login

from code.classes.Config import Config
from code.methods.evaluate_llm import evaluate_llm
from code.methods.select_model import select_model


def main():
    # Set huggingface access token for faster downloads.
    access_token = os.getenv("HF_TOKEN", None)

    if access_token is None:
        raise ValueError("Please set HF_TOKEN environment variable for huggingface faster download.")

    login(access_token)

    seed = 42
    model = "Qwen/Qwen2.5-7B-Instruct"
    model_name = model.split("/")[-1]
    dataset_type = "harmless"

    # Define the base_model.
    config = Config(model_alias=model_name, model_path=model)
    model_base = select_model(config=config)

    # Load state dict.
    state_dict_file = f"../../../../data/runs/state_dicts/{config.model_alias}/abliteration_state_dict_norm_preserving.pth"
    model_base.set_state_dict(torch.load(state_dict_file))

    # Create response from the original model
    evaluate_llm(
        harm_type=dataset_type,
        save_location_path=f"../../../../data/responses/{model}/",
        save_file_name=f"{dataset_type}_prompts_{model_name}_seed_{seed}.csv",
        model_base=model_base,
        config=config
    )


if __name__ == "__main__":
    main()
