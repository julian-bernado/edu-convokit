"""
play_engagement.py
Quick notebook / script helper for poking at Hugging-Face emotion models.

Core helpers
------------
load_model(model_name, top_k=None)
scan_column(df, text_col, model, labels, hist_dir=".", batch_size=32)
extreme(df, emotion, direction="top", k=5)

• No CLI parsing — just edit the variables in the `if __name__ == "__main__"` block.
• Produces per-label histogram PNGs (hist_<label>.png).
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import pipeline


# --------------------------------------------------------------------------- #
#  MODEL LOADING                                                              #
# --------------------------------------------------------------------------- #
def load_model(model_name: str, top_k: int | None = None):
    """
    Returns
    -------
    clf   : transformers.pipeline.TextClassificationPipeline
    labels: list[str]   order of labels returned by the model
    """
    clf = pipeline(
        "text-classification",
        model=model_name,
        top_k=top_k,
        truncation=True,
        return_all_scores=True,
    )
    labels = [d["label"] for d in clf("hello")[0]]
    return clf, labels


# --------------------------------------------------------------------------- #
#  SCORING + HISTOGRAMS                                                       #
# --------------------------------------------------------------------------- #
def scan_column(
    df: pd.DataFrame,
    text_col: str,
    model,
    labels: List[str],
    hist_dir: str | Path = ".",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    • Scores df[text_col] with `model`
    • Adds one float column per label
    • Saves hist_<label>.png into hist_dir
    • Returns the augmented dataframe
    """
    hist_dir = Path(hist_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)

    # ensure columns exist
    for lab in labels:
        df[lab] = np.nan

    # mini-batch inference
    for start in range(0, len(df), batch_size):
        chunk_series = df[text_col].iloc[start : start + batch_size]

        clean = (
            chunk_series               # pandas Series
            .fillna("")                # NaN/None → empty string
            .astype(str)               # ensure plain str
            .tolist()                  # now it's a list
        )

        outs = model(clean)            # list[list[dict(label,score)]]
        for df_idx, scores in zip(chunk_series.index, outs):
            for d in scores:
                df.at[df_idx, d["label"]] = d["score"]

    # histograms
    for lab in labels:
        data = df[lab].dropna()
        if data.empty:
            continue
        plt.figure()
        plt.hist(data, bins=30, edgecolor="black")
        plt.title(f"Histogram of {lab}")
        plt.xlabel("score")
        plt.ylabel("count")
        plt.tight_layout()
        out = hist_dir / f"hist_{lab}.png"
        plt.savefig(out)
        plt.close()
        print(f"[saved] {out}")

    return df


# --------------------------------------------------------------------------- #
#  EXTREME EXAMPLES                                                           #
# --------------------------------------------------------------------------- #
def extreme(
    df: pd.DataFrame,
    emotion: str,
    direction: Literal["top", "bottom"] = "top",
    k: int = 5,
) -> pd.DataFrame:
    """
    Return k rows with highest or lowest value for `emotion`.
    Requires that scan_column has created the column.
    """
    if emotion not in df.columns:
        raise ValueError(
            f"'{emotion}' column not found — run scan_column first."
        )
    asc = direction == "bottom"
    idx = (
        df[[emotion]]
        .dropna()
        .sort_values(emotion, ascending=asc)
        .head(k)
        .index
    )
    return df.loc[idx]


# --------------------------------------------------------------------------- #
#  SIMPLE DRIVER — EDIT THESE THREE LINES (Path-friendly)                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    PROJECT_ROOT = Path("/Users/jubernado/Google Drive/Shared drives/NSSA Research/MathBenchmarks/data/datafiles/measuredev_exploratory/engagement")
    DATA_PATH = PROJECT_ROOT.parent / "captioned_conversations.csv"

    TEXT_COL   = "text"
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"

    df = pd.read_csv(DATA_PATH)  # pandas accepts Path
    df = df[df["speaker"] == "student"]
    df = df.head(100)
    df["text"] = [str(txt) for txt in df["text"]]
    for txt in df["text"]:
        if not isinstance(txt, str):
            print("GRAHHHHH")
            print(txt)
    clf, lbls = load_model(MODEL_NAME)
    df = scan_column(df, text_col=TEXT_COL, model=clf, labels=lbls)

    # Example: top 5 most curious lines
    LABEL_COLS = [
        "pessimism",
        "joy",
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "love",
        "sadness",
        "surprise",
        "trust",
        "optimism"
    ]

    for lab in LABEL_COLS:
        # top-10
        top10 = (
            df[[TEXT_COL, lab]]
            .dropna()
            .sort_values(lab, ascending=False)
            .head(10)
        )
        print(f"\n===== TOP 10 {lab.upper()} =====")
        print(top10.to_markdown(index=False))

        # bottom-10
        bottom10 = (
            df[[TEXT_COL, lab]]
            .dropna()
            .sort_values(lab, ascending=True)
            .head(10)
        )
        print(f"\n----- BOTTOM 10 {lab.upper()} -----")
        print(bottom10.to_markdown(index=False))

        # Show top 10 and bottom 10 utterances across all emotion columns
        emotion_cols = [col for col in df.columns if col in lbls]

        # Top 10 utterances with highest value across any emotion
        df['max_emotion_score'] = df[emotion_cols].max(axis=1)
        df['max_emotion_label'] = df[emotion_cols].idxmax(axis=1)
        top10_any = df.sort_values('max_emotion_score', ascending=False).head(5)
        print("\n===== TOP 10 UTTERANCES (ANY EMOTION) =====")
        print(top10_any[[TEXT_COL, 'max_emotion_label', 'max_emotion_score']].to_markdown(index=False))

        # Bottom 10 utterances with lowest value across any emotion
        df['min_emotion_score'] = df[emotion_cols].min(axis=1)
        df['min_emotion_label'] = df[emotion_cols].idxmin(axis=1)
        bottom10_any = df.sort_values('min_emotion_score', ascending=True).head(5)
        print("\n----- BOTTOM 10 UTTERANCES (ANY EMOTION) -----")
        print(bottom10_any[[TEXT_COL, 'min_emotion_label', 'min_emotion_score']].to_markdown(index=False))