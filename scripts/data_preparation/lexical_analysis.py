
import pandas as pd
import matplotlib.pyplot as plt


class LexicalAnalysis:

    def analyze(self, model_df):
        results = []

        for label in model_df["label"].unique():
            texts = (
                model_df.loc[
                    model_df["label"] == label, "normalized_text"
                ]
                .fillna("")
                .astype(str)
            )

            total_words = 0
            vocabulary = set()

            for text in texts:
                words = text.split()
                total_words += len(words)
                vocabulary.update(words)

            vocabulary_size = len(vocabulary)

            results.append({
                "label": label,
                "total_words": total_words,
                "vocabulary_size": vocabulary_size,
                "lexical_diversity": (
                    vocabulary_size / total_words
                    if total_words > 0 else 0
                )
            })

        return pd.DataFrame(results)

    def plot(self, results):
        results.plot(
            x="label",
            y="lexical_diversity",
            kind="bar",
            legend=False,
            figsize=(8, 5)
        )

        plt.title("Lexical Diversity Comparison")
        plt.xlabel("Text Class")
        plt.ylabel("Vocabulary Size / Total Words")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
