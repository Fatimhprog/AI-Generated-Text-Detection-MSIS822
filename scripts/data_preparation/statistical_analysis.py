
import pandas as pd
from statistical_features import StatisticalFeatures


class StatisticalAnalysis:

    def analyze(self, model_df):
        extractor = StatisticalFeatures()

        features = model_df["normalized_text"].apply(
            extractor.extract_all
        )

        features_df = pd.DataFrame(features.tolist())
        features_df["label"] = model_df["label"].values

        summary = features_df.groupby("label")[
            [
                "avg_word_length",
                "avg_sentence_length",
                "ttr"
            ]
        ].mean()

        return features_df, summary

