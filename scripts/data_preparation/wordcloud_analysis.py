
import re
import os
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display


class WordCloudAnalysis:

    def plot_wordclouds(self, model_df):

        font_path = r"C:\Windows\Fonts\arial.ttf"

        if not os.path.exists(font_path):
            raise FileNotFoundError(
                "Arabic font not found. Check the font path."
            )

        for label in model_df["label"].unique():

            texts = model_df.loc[
                model_df["label"] == label,
                "normalized_text"
            ].dropna().astype(str)

            words = []

            for text in texts:
                words.extend(
                    re.findall(r"[\u0621-\u064A]+", text)
                )

            word_counts = Counter(words)

            if not word_counts:
                print(f"No Arabic words found for {label}")
                continue

            frequencies = {}

            for word, count in word_counts.items():
                reshaped_word = arabic_reshaper.reshape(word)
                display_word = get_display(reshaped_word)
                frequencies[display_word] = count

            cloud = WordCloud(
                font_path=font_path,
                width=1200,
                height=600,
                background_color="white",
                max_words=100
            ).generate_from_frequencies(frequencies)

            plt.figure(figsize=(12, 6))
            plt.imshow(cloud, interpolation="bilinear")
            plt.title(f"Word Cloud - {label}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
