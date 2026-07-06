from code.generate.plots.plot_llm_refusal_comparison import load_and_aggregate, plot_refusal_scores

CONFIG = {
    "qwen2.5-3B (harmful)": [
        "../../../data/responses/Qwen/Qwen2.5-3B-Instruct/harmful_prompts_Qwen2.5-3B-Instruct_seed_42.csv"
    ],
    "qwen2.5-3B (harmless)": [
        "../../../data/responses/Qwen/Qwen2.5-3B-Instruct/harmless_prompts_Qwen2.5-3B-Instruct_seed_42.csv"
    ],
    "qwen2.5-3B-abliterated - Standard (harmful)": [
        "../../../data/responses/Qwen/Qwen2.5-3B-Instruct-abliteration/standard/harmful_prompts_Qwen2.5-3B-Instruct_seed_42.csv"
    ],
    "qwen2.5-3B-abliterated - Standard (harmless)": [
        "../../../data/responses/Qwen/Qwen2.5-3B-Instruct-abliteration/standard/harmless_prompts_Qwen2.5-3B-Instruct_seed_42.csv"
    ],
    "qwen2.5-3B-abliterated - Norm Preserving (harmful)": [
        "../../../data/responses/Qwen/Qwen2.5-3B-Instruct-abliteration/norm/harmful_prompts_Qwen2.5-3B-Instruct_seed_42.csv"
    ],
    "qwen2.5-3B-abliterated - Norm Preserving (harmless)": [
        "../../../data/responses/Qwen/Qwen2.5-3B-Instruct-abliteration/norm/harmless_prompts_Qwen2.5-3B-Instruct_seed_42.csv"
    ]
}

OUTPUT_FILE_PATH = "../../../data/images/model_comparison/"
OUTPUT_FILE_NAME = "without_with_abliteration_Qwen_3B_comparison.png"


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
        "Vergleich ohne Abliteration vs. Abliteration-Methoden der Qwen-3B-Instruct"
    )


if __name__ == "__main__":
    main()
