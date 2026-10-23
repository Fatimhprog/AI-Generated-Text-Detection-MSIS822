
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


class TextVisualizer:

    def __init__(self, font_path="C:/Windows/Fonts/arial.ttf"):
        self.font_path = font_path

    def word_clouds(
        self,
        df,
        text_column="text",
        label_column="label",
        stop_words=None
    ):
        """
        Create a separate word cloud for each class.
        """
        stop_words = set(stop_words or [])

        labels = df[label_column].dropna().unique()

        for label in labels:
            texts = df.loc[
                df[label_column] == label, text_column
            ].dropna().astype(str)

            words = []

            for text in texts:
                words.extend(text.split())

            words = [
                word for word in words
                if word not in stop_words
            ]

            if not words:
                print(f"No words available for class: {label}")
                continue

            wordcloud = WordCloud(
                font_path=self.font_path,
                width=1200,
                height=600,
                background_color="white",
                collocations=False
            ).generate(" ".join(words))

            plt.figure(figsize=(14, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud - {label}")
            plt.show()

    def top_ngrams(
        self,
        df,
        n=2,
        top_n=15,
        text_column="text",
        label_column="label",
        stop_words=None
    ):
        """
        Display the most frequent n-grams for each class.
        n=1: unigram, n=2: bigram, n=3: trigram.
        """
        if n < 1:
            raise ValueError("n must be at least 1")

        stop_words = set(stop_words or [])

        results = {}

        for label in df[label_column].dropna().unique():
            texts = (
                df.loc[df[label_column] == label, text_column]
                .dropna()
                .astype(str)
            )

            if texts.empty:
                continue

            vectorizer = CountVectorizer(
                ngram_range=(n, n),
                stop_words=list(stop_words) if stop_words else None,
                token_pattern=r"(?u)\b\w+\b",
                max_features=50000
            )

            try:
                matrix = vectorizer.fit_transform(texts)
            except ValueError:
                print(f"No n-grams found for class: {label}")
                continue

            frequencies = matrix.sum(axis=0).A1
            ngrams = vectorizer.get_feature_names_out()

            top_indices = frequencies.argsort()[-top_n:][::-1]

            top_items = [
                (ngrams[i], int(frequencies[i]))
                for i in top_indices
            ]

            results[label] = top_items

            print(f"\nTop {top_n} {n}-grams - {label}")

            for phrase, count in top_items:
                print(f"{phrase}: {count}")

            phrases = [item[0] for item in top_items][::-1]
            counts = [item[1] for item in top_items][::-1]

            plt.figure(figsize=(12, 7))
            plt.barh(phrases, counts)
            plt.xlabel("Frequency")
            plt.ylabel("N-gram")
            plt.title(f"Top {top_n} {n}-grams - {label}")
            plt.tight_layout()
            plt.show()

        return results
