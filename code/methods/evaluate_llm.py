import gc
import os

import pandas as pd
from pandas import DataFrame

from code.classes.Config import Config
from code.classes.RefusalDetector import RefusalDetector
from code.classes.generators.BaseModel import BaseModel
from code.methods.setup_device import cleanup_gpu
from data.prompts.dataset.load_prompts import load_prompts


def print_percentage_of_response_types(df: DataFrame):
    # Anzahl pro Response-Type
    counts = df["response_type"].value_counts()

    # Prozentualer Anteil pro Response-Type
    percentages = (
        df["response_type"]
        .value_counts(normalize=True)
        .mul(100)
    )

    # Gemeinsame Tabelle erstellen
    response_type_stats = pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })

    print(response_type_stats)


def evaluate_llm(harm_type: str, save_location_path: str, save_file_name: str, model_base: BaseModel, config: Config):
    """
    Evaluate and creates responses from given prompts and a model. Then saves it in a csv file.
    :param config: The config to use.
    :param harm_type: The harm type of the prompts (harmful or harmless).
    :param save_location_path: The location to save the csv file.
    :param save_file_name: The csv filename.
    :param model_base: The model to use.
    :return: A list of times per prompt, the time for all prompts and the tokens for all prompts + responses.
    """

    # Load the prompts and put it into a list.
    prompts = load_prompts(n_samples=config.n_test, harm_type=harm_type, seed=config.seed, instructions_only=False)

    # Test all prompts and generate the responses.
    results = model_base.generate_multiple(prompts, batch_size=config.batch_size)

    # Evaluate the responses with the detector.
    detector = RefusalDetector()
    evaluated = detector.detect(results)

    df = pd.DataFrame(evaluated)

    print_percentage_of_response_types(df)

    # Save the results of the evaluation.
    if not os.path.exists(save_location_path):
        os.makedirs(save_location_path)
    df.to_csv(save_location_path + save_file_name, index=False)

    del detector.model, detector
    gc.collect()
    cleanup_gpu()
