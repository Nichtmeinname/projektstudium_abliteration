from code.generate.plots.method.plot_category_refusal_type_heatmap import heatmap_category_refusal_type

if __name__ == '__main__':
    heatmap_category_refusal_type(
        "Qwen2.5-3B-Instruct",
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct/Quantization/harmful_prompts_seed_42.csv",
        "../../../../data/images/heatmap/category_refusal_type/Qwen2.5-3B-Instruct/",
        "harmful_Qwen2.5-3B-Instruct.png"
    )

    heatmap_category_refusal_type(
        "Qwen2.5-3B-Instruct",
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct/Quantization/harmless_prompts_seed_42.csv",
        "../../../../data/images/heatmap/category_refusal_type/Qwen2.5-3B-Instruct/",
        "harmless_Qwen2.5-3B-Instruct.png"
    )

    heatmap_category_refusal_type(
        "Qwen2.5-3B-Instruct-Abliterated (Norm Preserving)",
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct_abliterated_norm_preserving/Quantization/harmful_prompts_seed_42.csv",
        "../../../../data/images/heatmap/category_refusal_type/Qwen2.5-3B-Instruct/",
        "harmful_Qwen2.5-3B-Instruct-Abliterated-Norm-Preserving.png"
    )

    heatmap_category_refusal_type(
        "Qwen2.5-3B-Instruct-Abliterated (Norm Preserving)",
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct_abliterated_norm_preserving/Quantization/harmless_prompts_seed_42.csv",
        "../../../../data/images/heatmap/category_refusal_type/Qwen2.5-3B-Instruct/",
        "harmless_Qwen2.5-3B-Instruct-Abliterated-Norm-Preserving.png"
    )

    heatmap_category_refusal_type(
        "Qwen2.5-3B-Instruct-Abliterated (Standard)",
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct_abliterated_standard/Quantization/harmful_prompts_seed_42.csv",
        "../../../../data/images/heatmap/category_refusal_type/Qwen2.5-3B-Instruct/",
        "harmful_Qwen2.5-3B-Instruct-Abliterated-Standard.png"
    )

    heatmap_category_refusal_type(
        "Qwen2.5-3B-Instruct-Abliterated (Standard)",
        "../../../../data/responses/Qwen/Qwen2.5-3B-Instruct_abliterated_standard/Quantization/harmless_prompts_seed_42.csv",
        "../../../../data/images/heatmap/category_refusal_type/Qwen2.5-3B-Instruct/",
        "harmless_Qwen2.5-3B-Instruct-Abliterated-Standard.png"
    )
