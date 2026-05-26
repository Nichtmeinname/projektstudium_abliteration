from code.methods.create_responses_csv import create_responses_csv


def main():
    seed = 42
    model = "Qwen/Qwen2.5-3B-Instruct"
    model_name = model.split("/")[-1]
    dataset_type = "harmless"

    # Create response from the original model
    create_responses_csv(
        f"../../../../data/prompts/{dataset_type}/test-00000-of-00001.parquet",
        f"../../../../data/responses/{model}/",
        f"{dataset_type}_prompts_{model_name}_seed_{seed}.csv",
        model,
        seed,
        False
    )


if __name__ == "__main__":
    main()
