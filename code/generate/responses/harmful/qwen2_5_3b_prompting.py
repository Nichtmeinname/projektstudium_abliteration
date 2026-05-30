from code.methods.evaluate_llm import evaluate_llm


def main():
    seed = 42
    model = "Qwen/Qwen2.5-3B-Instruct"
    model_name = model.split("/")[-1]
    dataset_type = "harmful"

    # Create response from the original model
    evaluate_llm(
        dataset_type,
        f"../../../../data/responses/{model}/",
        f"{dataset_type}_prompts_{model_name}_seed_{seed}.csv",
        model
    )


if __name__ == "__main__":
    main()
