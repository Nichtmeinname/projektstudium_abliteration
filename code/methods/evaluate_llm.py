import gc
import os

import pandas as pd
import torch
from huggingface_hub import login

from code.classes.Config import Config
from code.classes.RefusalDetector import RefusalDetector
from code.methods.select_model import select_model
from data.prompts.dataset.load_prompts import load_prompts


def evaluate_llm(harm_type: str, save_location_path: str, save_file_name: str, model: str):
    """
    Evaluate and creates responses from given prompts and a model. Then saves it in a csv file.
    :param harm_type: The harm type of the prompts (harmful or harmless).
    :param save_location_path: The location to save the csv file.
    :param save_file_name: The csv filename.
    :param model: The model to use.
    :return: A list of times per prompt, the time for all prompts and the tokens for all prompts + responses.
    """
    # Set huggingface access token for faster downloads.
    access_token = os.getenv("HF_TOKEN", None)

    if access_token is None:
        raise ValueError("Please set HF_TOKEN environment variable for huggingface faster download.")

    login(access_token)

    # Load the prompts and put it into a list.
    config = Config(model.split("/")[0], model)
    prompts = load_prompts(n_samples=config.n_test, harm_type=harm_type, seed=config.seed, instructions_only=True)

    # Define the base_model.
    base_model = select_model(config=config)

    # Test all prompts and generate the responses.
    results = base_model.generate_multiple(prompts, batch_size=config.batch_size)

    # Delete the base_model and clear the gpu cache, because detector also loads a model.
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Evaluate the responses with the detector.
    detector = RefusalDetector()
    evaluated = detector.detect(results)

    # Save the results of the evaluation.
    if not os.path.exists(save_location_path):
        os.makedirs(save_location_path)
    pd.DataFrame(evaluated).to_csv(save_location_path + save_file_name, index=False)
