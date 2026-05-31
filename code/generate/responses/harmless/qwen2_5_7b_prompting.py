from code.methods.evaluate_llm import evaluate_llm


def main():
    seed = 42
    model = "Qwen/Qwen2.5-7B-Instruct"
    model_name = model.split("/")[-1]
    dataset_type = "harmless"

    # Create response from the original model
    evaluate_llm(
        harm_type=dataset_type,
        save_location_path=f"../../../../data/responses/{model}/",
        save_file_name=f"{dataset_type}_prompts_{model_name}_seed_{seed}.csv",
        model=model
    )


if __name__ == "__main__":
    main()
