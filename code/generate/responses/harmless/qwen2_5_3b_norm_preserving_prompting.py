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
    dataset_type = "harmless"
    model_dir = f"../../../../data/runs/models/Qwen2.5-3B-Instruct/Qwen2.5-3B-Instruct_abliterated_norm_preserving"
    model_alias = model_dir.split("/")[-1]

    # Define the base_model.
    config = Config(model_alias=model_alias, model_path=model_dir)
    model_base = select_model(config=config)
    quantization_used = "Quantization" if config.four_bit_quantization else "NoQuantization"

    # Create response from the original model
    evaluate_llm(
        harm_type=dataset_type,
        save_location_path=f"../../../../data/responses/Qwen/{config.model_alias}/{quantization_used}/",
        save_file_name=f"{dataset_type}_prompts_seed_{seed}.csv",
        model_base=model_base,
        config=config
    )


if __name__ == "__main__":
    main()
