from pathlib import Path
from typing import List, Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from edu_convokit.annotation import Annotator

PROJECT_ROOT = Path("/Users/jubernado/Google Drive/Shared drives/NSSA Research/MathBenchmarks/data/datafiles/measuredev_exploratory/engagement")
DATA_PATH = PROJECT_ROOT.parent / "captioned_conversations.csv"
df = pd.read_csv(DATA_PATH)
print(df.head())
df["text"] = [str(txt) for txt in df["text"]]

# load in an annotator
ant = Annotator()
df = ant.get_engagement(
    df,
    time_column="timestamp",
    speaker_column="speaker",
    speaker_values="student",
    transcript_column="filename",
    event_type_column="type",
    text_column="text"
)
print(df.head())

def plot_engagement_for_filename(df, filename=None):
    if filename is None:
        filename = np.random.choice(df["filename"].unique())
        print(f"Randomly selected filename: {filename}")
    filtered = df[df["filename"] == filename].reset_index(drop=True)
    if filtered.empty:
        print(f"No data found for filename: {filename}")
        return
    plt.figure(figsize=(12, 5))

    # Plot engagement and avg_engagement by index (subplot 1)
    plt.subplot(1, 2, 1)
    plt.plot(filtered.index, filtered["engagement"], marker='o', label='Engagement')
    plt.title(f'Engagement by Index\nFilename: {filename}')
    plt.xlabel('Index')
    plt.ylabel('Engagement')
    plt.legend()

    # Plot engagement and avg_engagement by index (subplot 1)
    plt.subplot(1, 2, 2)
    plt.plot(filtered.index, filtered["avg_engagement"], marker='o', label='avg_engagement')
    plt.title(f'Engagement by Index\nFilename: {filename}')
    plt.xlabel('Index')
    plt.ylabel('Engagement')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
plot_engagement_for_filename(df)

k = 5  # Set the number of transcripts you want
last_entries = df.sort_values("timestamp").groupby("filename").tail(1)
top_k = last_entries.nlargest(k, "avg_engagement")
print("Top k filenames with highest last-entry engagement:")
for fname in top_k["filename"]:
    print(fname)