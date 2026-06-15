import argparse
import os

from huggingface_hub import login

from code.classes.Config import Config
from code.methods.evaluate_llm import evaluate_llm
from code.methods.find_best_direction.select_most_effective_refusal_direction import \
    select_most_effective_refusal_direction
from code.methods.mean_diff_methods.generate_mean_diff import generate_mean_diff
from code.methods.modify_weights.apply_abliteration import apply_abliteration
from code.methods.select_model import select_model
from data.prompts.dataset.load_prompts import load_prompts


def parse_arguments() -> str:
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path as argument.")
    parser.add_argument('--model', type=str, required=True, help='Huggingface model path.')
    return parser.parse_args().model


def run_pipeline(model_path: str):
    config = Config(model_alias=model_path.split("/")[-1], model_path=model_path)
    print("Model: ", model_path)

    # Set huggingface access token for faster downloads.
    access_token = os.getenv("HF_TOKEN", None)

    if access_token is None:
        raise ValueError("Please set HF_TOKEN environment variable for huggingface faster download.")

    login(access_token)
    model_base = select_model(config)

    # Load lists of train and validation sets
    harmful_train = load_prompts(n_samples=config.n_train, harm_type="harmful",
                                 dataset_type="train", seed=config.seed, instructions_only=True)
    harmful_val = load_prompts(n_samples=config.n_val, harm_type="harmful",
                               dataset_type="val", seed=config.seed, instructions_only=True)
    harmless_train = load_prompts(n_samples=config.n_train, harm_type="harmless",
                                  dataset_type="train", seed=config.seed, instructions_only=True)
    harmless_val = load_prompts(n_samples=config.n_val, harm_type="harmless",
                                dataset_type="val", seed=config.seed, instructions_only=True)

    # 1. Generate the mean direction between harmless and harmful prompts
    print("1. Start generating mean diff between harmful and harmless train prompts...")
    mean_diffs = generate_mean_diff(config, model_base, harmful_train, harmless_train)

    print("    Generation of mean diff succeeded.")
    print("2. Select the most effective refusal direction...")
    pos, layer, direction = select_most_effective_refusal_direction(config, model_base, harmful_val, harmless_val,
                                                                    mean_diffs)

    print(f"    Found best refusal direction in token position {pos} and layer {layer}: ", direction)

    print("3. Apply abliteration and save models state dict...")
    state_dict = apply_abliteration(config=config,
                                    model_base=model_base,
                                    refusal_direction=direction)

    dataset_type = "harmful"
    model_base.set_state_dict(state_dict)
    print("4. Applied abliteration and evaluate on harmful prompts...")

    evaluate_llm(
        harm_type=dataset_type,
        save_location_path=f"../../data/responses/Qwen/{config.model_alias}-abliteration/",
        save_file_name=f"{dataset_type}_prompts_{config.model_alias}_seed_{config.seed}.csv",
        model_base=model_base,
        config=config
    )

    dataset_type = "harmless"
    print("5. Applied abliteration and evaluate on harmless prompts...")

    evaluate_llm(
        harm_type=dataset_type,
        save_location_path=f"../../data/responses/Qwen/{config.model_alias}-abliteration/",
        save_file_name=f"{dataset_type}_prompts_{config.model_alias}_seed_{config.seed}.csv",
        model_base=model_base,
        config=config
    )


if __name__ == "__main__":
    model_ = parse_arguments()
    run_pipeline(model_)
