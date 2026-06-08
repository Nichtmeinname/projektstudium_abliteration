import argparse
import os

from huggingface_hub import login

from code.classes.Config import Config
from code.methods.find_best_direction.select_most_effective_refusal_direction import \
    select_most_effective_refusal_direction
from code.methods.mean_diff_methods.generate_mean_diff import generate_mean_diff
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
    model = select_model(config)

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
    mean_diffs = generate_mean_diff(config, model, harmful_train, harmless_train)

    print("    Generation of mean diff succeeded.")
    print("2. Select the most effective refusal direction...")
    pos, layer, direction = select_most_effective_refusal_direction(config, model, harmful_val, harmless_val,
                                                                    mean_diffs)

    print(f"    Found best refusal direction in token position {pos} and layer {layer}: ", direction)


if __name__ == "__main__":
    model_ = parse_arguments()
    run_pipeline(model_)
