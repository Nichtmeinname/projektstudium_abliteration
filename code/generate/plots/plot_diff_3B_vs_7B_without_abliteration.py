from code.generate.plots.plot_refusal_scores import plot_refusal_scores, load_and_aggregate

CONFIG = {
    "qwen2.5-3B (Unethische Anfragen)": [
        "../../../data/responses/Qwen/Qwen2.5-3B-Instruct/harmful_prompts_Qwen2.5-3B-Instruct_seed_42.csv"
    ],
    "qwen2.5-3B (Harmlose Anfragen)": [
        "../../../data/responses/Qwen/Qwen2.5-3B-Instruct/harmless_prompts_Qwen2.5-3B-Instruct_seed_42.csv"
    ],
    "qwen2.5-7B (Unethische Anfragen)": [
        "../../../data/responses/Qwen/Qwen2.5-7B-Instruct/harmful_prompts_Qwen2.5-7B-Instruct_seed_42.csv"
    ],
    "qwen2.5-7B (Harmlose Anfragen)": [
        "../../../data/responses/Qwen/Qwen2.5-7B-Instruct/harmless_prompts_Qwen2.5-7B-Instruct_seed_42.csv"
    ]
}

OUTPUT_FILE_PATH = "../../../data/images/model_comparison/"
OUTPUT_FILE_NAME = "3B_vs_7B_without_abliteration.png"


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
        "Vergleich Qwen 3B vs 7B Modell ohne Abliteration\n n=" + str(
            sum(aggregated[next(iter(aggregated))].values)
        ) + " Anfragen",
        bar_width=0.2
    )


if __name__ == "__main__":
    main()
