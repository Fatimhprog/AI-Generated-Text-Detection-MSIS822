
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer


class Visualization:

    def plot_statistics(self, summary):
        summary.plot(kind="bar", figsize=(10, 5))

        plt.title("Statistical Features Comparison")
        plt.xlabel("Text Class")
        plt.ylabel("Average Value")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_top_words(self, model_df, n=10):
        for label in model_df["label"].unique():
            texts = model_df.loc[
                model_df["label"] == label, "normalized_text"
            ]

            words = []

            for text in texts:
                words.extend(re.findall(r"\w+", str(text)))

            top_words = Counter(words).most_common(n)

            word_list = [item[0] for item in top_words]
            frequencies = [item[1] for item in top_words]

            plt.figure(figsize=(10, 5))
            plt.bar(word_list, frequencies)

            plt.title(f"Top {n} Words - {label}")
            plt.xlabel("Words")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def plot_ngrams(self, model_df, n=2, top_k=10):
        for label in model_df["label"].unique():
            texts = model_df.loc[
                model_df["label"] == label, "normalized_text"
            ].fillna("").astype(str)

            vectorizer = CountVectorizer(
                ngram_range=(n, n),
                token_pattern=r"(?u)\b\w+\b"
            )

            try:
                matrix = vectorizer.fit_transform(texts)
            except ValueError:
                print(f"No n-grams found for {label}")
                continue

            frequencies = matrix.sum(axis=0).A1
            phrases = vectorizer.get_feature_names_out()
            order = frequencies.argsort()[::-1][:top_k]

            plt.figure(figsize=(10, 5))
            plt.bar(phrases[order], frequencies[order])

            plt.title(f"Top {top_k} {n}-word Phrases - {label}")
            plt.xlabel("Phrases")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
