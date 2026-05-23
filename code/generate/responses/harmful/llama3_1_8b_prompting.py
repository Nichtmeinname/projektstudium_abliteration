from code.methods.create_responses_csv import create_responses_csv


def main():
    seed = 42
    model_name = "llama3.1"
    model_param = "8b"
    dataset_type = "harmful"

    # Create response from the original model
    create_responses_csv(
        f"../../../../data/prompts/{dataset_type}/test-00000-of-00001.parquet",
        f"../../../../data/responses/{model_name}/",
        f"{dataset_type}_prompts_ollama_{model_name}_seed_{seed}.csv",
        f"{model_name}:{model_param}",
        seed
    )


if __name__ == "__main__":
    main()
