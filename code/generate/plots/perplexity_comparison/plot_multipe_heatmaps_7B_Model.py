from code.generate.plots.method.plot_perplexity_heatmaps import plot_perplexity_heatmaps

if __name__ == "__main__":
    model_name = "Qwen2.5-7B-Instruct"
    weights_modified = ""

    plot_perplexity_heatmaps(
        f"../../../../data/responses/Qwen/{model_name}/Quantization/harmful_prompts_seed_42.csv",
        f"Kategorie-Perplexity-Tabelle für {model_name}{weights_modified}\nUnethische Anfragen",
        f"Response-Perplexity-Tabelle für {model_name}{weights_modified}\nUnethische Anfragen",
        f"../../../../data/images/heatmap/category_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmful.png",
        f"../../../../data/images/heatmap/response_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmful.png"
    )

    plot_perplexity_heatmaps(
        f"../../../../data/responses/Qwen/{model_name}/Quantization/harmless_prompts_seed_42.csv",
        f"Kategorie-Perplexity-Tabelle für {model_name}{weights_modified}\nHarmlose Anfragen",
        f"Response-Perplexity-Tabelle für {model_name}{weights_modified}\nHarmlose Anfragen",
        f"../../../../data/images/heatmap/category_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmless.png",
        f"../../../../data/images/heatmap/response_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmless.png"
    )

    model_name = "Qwen2.5-7B-Instruct_abliterated_norm_preserving"
    weights_modified = "_Modified_Attn_O___MLP_DOWN"

    plot_perplexity_heatmaps(
        f"../../../../data/responses/Qwen/{model_name}/Quantization/Modified_Self_Attn_O___MLP_DOWN/harmful_prompts_seed_42.csv",
        f"Kategorie-Perplexity-Tabelle für {model_name}{weights_modified}\nUnethische Anfragen",
        f"Response-Perplexity-Tabelle für {model_name}{weights_modified}\nUnethische Anfragen",
        f"../../../../data/images/heatmap/category_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmful.png",
        f"../../../../data/images/heatmap/response_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmful.png"
    )

    plot_perplexity_heatmaps(
        f"../../../../data/responses/Qwen/{model_name}/Quantization/Modified_Self_Attn_O___MLP_DOWN/harmless_prompts_seed_42.csv",
        f"Kategorie-Perplexity-Tabelle für {model_name}{weights_modified}\nHarmlose Anfragen",
        f"Response-Perplexity-Tabelle für {model_name}{weights_modified}\nHarmlose Anfragen",
        f"../../../../data/images/heatmap/category_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmless.png",
        f"../../../../data/images/heatmap/response_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmless.png"
    )

    model_name = "Qwen2.5-7B-Instruct_abliterated_standard"
    weights_modified = "_Modified_Attn_O___MLP_DOWN"

    plot_perplexity_heatmaps(
        f"../../../../data/responses/Qwen/{model_name}/Quantization/Modified_Self_Attn_O___MLP_DOWN/harmful_prompts_seed_42.csv",
        f"Kategorie-Perplexity-Tabelle für {model_name}{weights_modified}\nUnethische Anfragen",
        f"Response-Perplexity-Tabelle für {model_name}{weights_modified}\nUnethische Anfragen",
        f"../../../../data/images/heatmap/category_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmful.png",
        f"../../../../data/images/heatmap/response_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmful.png"
    )

    plot_perplexity_heatmaps(
        f"../../../../data/responses/Qwen/{model_name}/Quantization/Modified_Self_Attn_O___MLP_DOWN/harmless_prompts_seed_42.csv",
        f"Kategorie-Perplexity-Tabelle für {model_name}{weights_modified}\nHarmlose Anfragen",
        f"Response-Perplexity-Tabelle für {model_name}{weights_modified}\nHarmlose Anfragen",
        f"../../../../data/images/heatmap/category_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmless.png",
        f"../../../../data/images/heatmap/response_perplexity/{model_name}/",
        f"{model_name}{weights_modified}_harmless.png"
    )
