import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def load_and_aggregate(file_paths: list[str]):
    """Alle CSV-Dateien eines LLMs einlesen und response_type zusammenzählen."""
    frames = []
    for path in file_paths:
        p = Path(path)
        if not p.exists():
            print(f"Datei nicht gefunden, wird übersprungen: {path}")
            continue
        df = pd.read_csv(p)
        if "response_type" not in df.columns:
            raise ValueError(f"Spalte 'response_type' fehlt in: {path}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError("Keine CSV-Dateien konnten geladen werden.")

    combined = pd.concat(frames, ignore_index=True)
    return combined["response_type"].value_counts().sort_index()


def plot_refusal_scores(data: dict[str, pd.Series], output_file_path: str, output_file_name: str, title: str):
    """Gruppiertes Balkendiagramm für alle LLMs und response_types erstellen."""

    # Alle vorkommenden response_types sammeln
    all_types = ["No Refusal", "Refusal Unethical", "Disclaimer Unethical", "Refusal Capability",
                 "Disclaimer Capability"]

    llm_names = list(data.keys())
    n_llms = len(llm_names)
    n_types = len(all_types)

    x = np.arange(n_types)
    bar_width = 0.1
    offsets = np.linspace(
        -(n_llms - 1) / 2 * bar_width,
        (n_llms - 1) / 2 * bar_width,
        n_llms,
    )

    colors = ["#4C72B0", "#DD8452", "#55A868", "#00ffff", "#9a32cd", "#ff1493",
              "#ffff00", "#ff0000"]  # Blue, Orange, Green, Aqua, Violet, Pink, Yellow, Red

    fig, ax = plt.subplots(figsize=(max(10, n_types * 2), 6))

    for i, (llm_name, offset) in enumerate(zip(llm_names, offsets)):
        counts = [data[llm_name].get(rt, 0) for rt in all_types]
        bars = ax.bar(
            x + offset,
            counts,
            width=bar_width,
            label=llm_name,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.6,
        )
        # Wert über jedem Balken anzeigen
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    str(val),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=25, ha="right", fontsize=11)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Response Type", fontsize=13, labelpad=8)
    ax.set_ylabel("Anzahl", fontsize=13, labelpad=8)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.legend(title="LLM", fontsize=11, title_fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    save_location = os.path.join(output_file_path, output_file_name)
    plt.savefig(save_location, dpi=300)
    print(f"\nDiagramm gespeichert: {save_location}")
    plt.show()
