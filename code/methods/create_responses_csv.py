import gc
import os
import random

import pandas as pd
import torch
from huggingface_hub import login

from code.classes.LLMCustomGenerator import LLMCustomGenerator
from code.classes.ParquetProbe import ParquetProbe
from code.classes.RefusalDetector import RefusalDetector


def create_responses_csv(parquet_file: str, save_location_path: str, save_file_name: str, model: str, seed: int,
                         full_parquet_file: bool = True):
    """
    Creates responses from given prompts and a model. Then saves it in a csv file.
    :param parquet_file: The parquet file with the corresponding prompts.
    :param save_location_path: The location to save the csv file.
    :param save_file_name: The csv filename.
    :param model: The model to use.
    :param seed: The seed for the model.
    :param full_parquet_file: Whether to use full parquet file or not (Default: True).
    :return: A list of times per prompt, the time for all prompts and the tokens for all prompts + responses.
    """
    random.seed(seed)
    # Set huggingface access token for faster downloads.
    access_token = os.getenv("HF_TOKEN", None)

    if access_token is None:
        raise ValueError("Please set HF_TOKEN environment variable for huggingface faster download.")

    login(access_token)

    # Load the harmful prompts and put it into a list.
    df = pd.read_parquet(parquet_file)
    prompts = df["text"].tolist()
    random.shuffle(prompts)
    prompts = prompts if full_parquet_file else prompts[:104]

    # Define the generator, probe and detector.
    generator = LLMCustomGenerator(model_name=model, seed=seed)
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
