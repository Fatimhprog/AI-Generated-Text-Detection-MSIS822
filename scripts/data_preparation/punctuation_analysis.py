
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re


class PunctuationAnalysis:

    def analyze(self, model_df):
        punctuation_marks = [
            ".", ",", "،", ";", "؛", ":", "؟",
            "!", "(", ")", "[", "]", '"', "'",
            "-", "—", "…"
        ]

        results = []

        for label in model_df["label"].unique():
            texts = (
                model_df.loc[
                    model_df["label"] == label, "raw_text"
                ]
                .fillna("")
                .astype(str)
            )

            counts = Counter()

            for text in texts:
                counts.update(
                    mark for mark in punctuation_marks
                    if mark in text
                    for _ in range(text.count(mark))
                )

            total_chars = sum(len(text) for text in texts)

            for mark in punctuation_marks:
                results.append({
                    "label": label,
                    "punctuation": mark,
                    "frequency": counts[mark],
                    "per_1000_chars": (
                        counts[mark] / total_chars * 1000
                        if total_chars > 0 else 0
                    )
                })

        return pd.DataFrame(results)

    def plot(self, results):
        pivot = results.pivot(
            index="punctuation",
            columns="label",
            values="per_1000_chars"
        )

        pivot.plot(kind="bar", figsize=(12, 6))
        plt.title("Punctuation Usage Comparison")
        plt.xlabel("Punctuation Marks")
        plt.ylabel("Occurrences per 1000 Characters")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
