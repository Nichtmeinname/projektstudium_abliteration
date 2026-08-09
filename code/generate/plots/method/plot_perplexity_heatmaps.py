import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_perplexity_heatmaps(csv_file: str,
                             title_category_perplexity: str,
                             title_response_perplexity: str,
                             save_location1: str,
                             file_name1: str,
                             save_location2: str,
                             file_name2: str):
    # CSV laden
    df = pd.read_csv(csv_file)

    # Sicherstellen, dass Perplexity numerisch ist
    df["perplexity"] = pd.to_numeric(
        df["perplexity"],
        errors="coerce"
    )

    # Ungültige Perplexity-Werte entfernen
    df = df.dropna(subset=["perplexity"])

    # ---------------------------------------------------------
    # Allgemeine Perplexity-Statistiken
    # ---------------------------------------------------------

    print("---------------------" * 5)
    print("Perplexity Statistics ", file_name1)
    print("---------------------")
    print(f"Mean: {df['perplexity'].mean():.4f}")
    print(f"Min:  {df['perplexity'].min():.4f}")
    print(f"Max:  {df['perplexity'].max():.4f}")
    print("---------------------" * 5)

    # ---------------------------------------------------------
    # Fehlende Kategorien sichtbar behalten
    # ---------------------------------------------------------

    df["category"] = df["category"].fillna("None")
    df["response_type"] = df["response_type"].fillna("None")

    # ---------------------------------------------------------
    # Mean Perplexity pro Category
    # ---------------------------------------------------------

    category_perplexity = (
        df.groupby("category")["perplexity"]
        .mean()
        .sort_values()
        .to_frame(name="Perplexity Mittelwert")
    )

    # ---------------------------------------------------------
    # Mean Perplexity pro Response Type
    # ---------------------------------------------------------

    response_order = [
        "No Refusal",
        "Disclaimer Unethical",
        "Disclaimer Capability",
        "Refusal Capability",
        "Refusal Unethical"
    ]

    response_type_perplexity = (
        df.groupby("response_type")["perplexity"]
        .mean()
        .reindex(response_order)
        .dropna()
        .to_frame(name="Perplexity Mittelwert")
    )

    # ---------------------------------------------------------
    # Heatmap: Category
    # ---------------------------------------------------------

    plt.figure(
        figsize=(6, max(4, len(category_perplexity) * 0.5))
    )

    sns.heatmap(
        category_perplexity,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5
    )

    plt.title(title_category_perplexity)
    plt.xlabel("")
    plt.ylabel("Kategorie")
    plt.tight_layout()
    if not os.path.exists(save_location1):
        os.makedirs(save_location1)
    plt.savefig(
        os.path.join(save_location1, file_name1),
        dpi=300,
        bbox_inches="tight"
    )

    # ---------------------------------------------------------
    # Heatmap: Response Type
    # ---------------------------------------------------------

    plt.figure(
        figsize=(6, max(4, len(response_type_perplexity) * 0.6))
    )

    sns.heatmap(
        response_type_perplexity,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5
    )

    plt.title(title_response_perplexity)
    plt.xlabel("")
    plt.ylabel("Response Type")
    plt.tight_layout()
    if not os.path.exists(save_location2):
        os.makedirs(save_location2)
    plt.savefig(
        os.path.join(save_location2, file_name2),
        dpi=300,
        bbox_inches="tight"
    )
