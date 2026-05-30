import gc
import os
import random

import pandas as pd
import torch
from huggingface_hub import login

from code.classes.Config import Config
from code.classes.LLMCustomGenerator import LLMCustomGenerator
from code.classes.ParquetProbe import ParquetProbe
from code.classes.RefusalDetector import RefusalDetector
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
    config = Config("Qwen", model)
    prompts = load_prompts(n_samples=config.n_test, harm_type=harm_type, seed=config.seed, instructions_only=True)

    # Define the generator, probe and detector.
    generator = LLMCustomGenerator(model_name=model, seed=config.seed, set_four_bit_quantisation=config.four_bit_quantization)
    probe = ParquetProbe(prompts=prompts)

    # Test all prompts and generate the responses.
    results, times_per_prompt, time_all_prompts, tokens = probe.probe(generator)

    # Delete the generator (model) and clear the gpu cache, because detector also loads a model.
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    detector = RefusalDetector()

    # Evaluate the responses with the detector.
    evaluated = detector.detect(results)

    # Save the results of the evaluation.
    if not os.path.exists(save_location_path):
        os.makedirs(save_location_path)
    pd.DataFrame(evaluated).to_csv(save_location_path + save_file_name, index=False)

    return times_per_prompt, time_all_prompts, tokens
