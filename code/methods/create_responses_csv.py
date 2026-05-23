import os

import pandas as pd

from code.classes.OllamaGenerator import OllamaGenerator
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
    # Load the harmful prompts and put it into a list.
    df = pd.read_parquet(parquet_file)
    prompts = df["text"].tolist() if full_parquet_file else df["text"].tolist()[:104]

    # Define the generator, probe and detector.
    generator = OllamaGenerator(model=model, seed=seed)
    probe = ParquetProbe(prompts=prompts)
    detector = RefusalDetector()

    # Test all prompts and generate the responses.
    results, times_per_prompt, time_all_prompts, tokens = probe.probe(generator)

    # Evaluate the responses with the detector.
    evaluated = []
    for r in results:
        flagged, score = detector.detect(r)

        evaluated.append({
            **r,
            "response_type": flagged,
            "accuracy_score": score
        })

    # Save the results of the evaluation.
    if not os.path.exists(save_location_path):
        os.makedirs(save_location_path)
    pd.DataFrame(evaluated).to_csv(save_location_path + save_file_name, index=False)

    return times_per_prompt, time_all_prompts, tokens
