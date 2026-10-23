
import pandas as pd
import matplotlib.pyplot as plt


class FunctionWordsAnalysis:

    def analyze(self, model_df):

        function_words = [
            "من", "في", "على", "إلى", "عن",
            "أن", "إن", "كان", "كانت", "هذا",
            "هذه", "الذي", "التي", "ما", "لا",
            "لم", "لن", "هو", "هي", "و"
        ]

        results = []

        for label in model_df["label"].unique():

            texts = model_df.loc[
                model_df["label"] == label,
                "normalized_text"
            ].fillna("").astype(str)

            all_words = " ".join(texts).split()
            total_words = len(all_words)

            for word in function_words:
                count = all_words.count(word)

                percentage = (
                    count / total_words * 100
                    if total_words > 0 else 0
                )

                results.append({
                    "label": label,
                    "word": word,
                    "frequency": count,
                    "percentage": percentage
                })

        return pd.DataFrame(results)

    def plot(self, results):

        pivot = results.pivot(
            index="word",
            columns="label",
            values="percentage"
        )

        pivot.plot(kind="bar", figsize=(12, 6))

        plt.title("Function Words Comparison")
        plt.xlabel("Arabic Function Words")
        plt.ylabel("Percentage of All Words")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
