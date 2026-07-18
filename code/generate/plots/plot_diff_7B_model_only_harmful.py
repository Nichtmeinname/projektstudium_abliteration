from code.generate.plots.plot_refusal_scores import load_and_aggregate, plot_refusal_scores

CONFIG = {
    "qwen2.5-7B (Unethische Anfragen)": [
        "../../../data/responses/Qwen/Qwen2.5-7B-Instruct/harmful_prompts_Qwen2.5-7B-Instruct_seed_42.csv"
    ],
    "qwen2.5-7B-abliterated - Standard (Unethische Anfragen)": [
        "../../../data/responses/Qwen/Qwen2.5-7B-Instruct-abliteration/standard/harmful_prompts_Qwen2.5-7B-Instruct_seed_42.csv"
    ],
    "qwen2.5-7B-abliterated - Norm Preserving (Unethische Anfragen)": [
        "../../../data/responses/Qwen/Qwen2.5-7B-Instruct-abliteration/norm/harmful_prompts_Qwen2.5-7B-Instruct_seed_42.csv"
    ]
}

OUTPUT_FILE_PATH = "../../../data/images/model_comparison/"
OUTPUT_FILE_NAME = "without_with_abliteration_Qwen_7B_only_harmful_comparison.png"


def main():
    print("Lade CSV-Dateien …")
    aggregated = {}
    for llm_name, paths in CONFIG.items():
        print(f"\n  {llm_name}")
        counts = load_and_aggregate(paths)
        aggregated[llm_name] = counts
        print(counts.to_string())

    plot_refusal_scores(
        aggregated,
        OUTPUT_FILE_PATH,
        OUTPUT_FILE_NAME,
        "Vergleich Qwen-7B-Instruct (nur Unethische Anfragen):\n ohne Abliteration vs. Abliteration-Methoden\n n=" + str(
            sum(aggregated[
                    next(iter(aggregated))
                ].values)
        ) + " Anfragen",
        bar_width=0.2
    )


if __name__ == "__main__":
    main()
