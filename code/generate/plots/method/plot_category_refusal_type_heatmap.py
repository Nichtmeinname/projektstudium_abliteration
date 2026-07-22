import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def heatmap_category_refusal_type(model_name: str, file_path: str, save_path: str, save_file_name: str):
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)

    response_order = [
        "No Refusal",
        "Disclaimer Unethical",
        "Disclaimer Capability",
        "Refusal Capability",
        "Refusal Unethical"
    ]

    heatmap_data = pd.crosstab(
        df["category"],
        df["response_type"]
    )

    heatmap_data = heatmap_data.reindex(
        columns=response_order,
        fill_value=0
    )

    plt.figure(figsize=(14, 10))

    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt="d",
        cmap="crest",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": "Anzahl"
        }
    )

    plt.xlabel("Response Type", fontsize=16)
    plt.ylabel("Kategorie", fontsize=16)

    plt.xticks(
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=12
    )

    plt.yticks(
        rotation=0,
        fontsize=12
    )

    for text in ax.texts:
        text.set_fontsize(11)

    harm_type = "Unethische" if "harmful" in file_path else "Harmlose"
    plt.title(
        f"Verteilung der Response Types pro Kategorie\n{harm_type} Anfragen\nModell: {model_name}",
        fontsize=20,
        weight="bold",
        pad=20
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(
        os.path.join(save_path, save_file_name),
        dpi=300,
        bbox_inches="tight"
    )
    plt.tight_layout()
    plt.show()
