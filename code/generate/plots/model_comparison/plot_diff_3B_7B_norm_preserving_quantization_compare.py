from code.generate.plots.method.plot_refusal_scores import plot_refusal_scores, load_and_aggregate

CONFIG = {
    "qwen2.5-3B-Abliterated - Norm Preserving  (Unethische Anfragen)\nNo Quantization": [
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct_abliterated_norm_preserving/NoQuantization/harmful_prompts_seed_42.csv"
    ],
    "qwen2.5-3B-Abliterated - Norm Preserving  (Unethische Anfragen)\nQuantization": [
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct_abliterated_norm_preserving/Quantization/harmful_prompts_seed_42.csv"
    ],
    "qwen2.5-7B-Abliterated - Norm Preserving  (Unethische Anfragen)\nNo Quantization": [
        "../../../../data/responses/Qwen/Qwen2.5-7B-Instruct_abliterated_norm_preserving/NoQuantization/harmful_prompts_seed_42.csv"
    ],
    "qwen2.5-7B-Abliterated - Norm Preserving  (Unethische Anfragen)\nQuantization": [
        "../../../../data/responses/Qwen/Qwen2.5-7B-Instruct_abliterated_norm_preserving/Quantization/harmful_prompts_seed_42.csv"
    ]
}

OUTPUT_FILE_PATH = "../../../../data/images/model_comparison/"
OUTPUT_FILE_NAME = "3B_vs_7B_abliterated_norm_preserving_quantization_compare.png"


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
        "Vergleich Qwen 3B/7B Modell-Abliterated-Norm-Preserving"
        "\nNo Quantization vs Quantization (n=" +
        str(
            sum(aggregated[next(iter(aggregated))].values)
        ) + ") Anfragen",
        bar_width=0.2
    )


if __name__ == "__main__":
    main()
