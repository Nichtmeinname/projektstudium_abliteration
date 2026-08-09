from code.generate.plots.method.plot_refusal_scores import load_and_aggregate, plot_refusal_scores

CONFIG = {
    "qwen2.5-3B-Abliterated - Modified W_O, MLP_DOWN": [
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct_abliterated_norm_preserving/Quantization/Modified_Self_Attn_O___MLP_DOWN/harmless_prompts_seed_42.csv"
    ],
    "qwen2.5-3B-Abliterated - Modified Attention Q, K, V, O, MLP UP, GATE, DOWN": [
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct_abliterated_norm_preserving/Quantization/Modified_Self_Attn_Q_K_V_O___MLP_DOWN_GATE_UP/harmless_prompts_seed_42.csv"
    ],
    "qwen2.5-3B-Abliterated - Modified Attention Q, K, V, MLP UP, GATE": [
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct_abliterated_norm_preserving/Quantization/Modified_Self_Attn_Q_K_V___MLP_GATE_UP/harmless_prompts_seed_42.csv"
    ]
}

OUTPUT_FILE_PATH = "../../../../data/images/model_comparison/"
OUTPUT_FILE_NAME = "abliteration_norm_preserving_different_modified_weights_Qwen_3B_only_harmless_comparison.png"


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
        "Vergleich Qwen-3B-Instruct-Abliterated-Norm-Preserving\n(nur Harmlose Anfragen): Unterschiedlich modifizierte Gewichte\n n=" + str(
            sum(aggregated[
                    next(iter(aggregated))
                ].values)
        ) + " Anfragen",
        bar_width=0.2
    )


if __name__ == "__main__":
    main()
