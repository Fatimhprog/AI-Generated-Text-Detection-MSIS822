
import re
import nltk
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords

# Download Arabic stopwords if needed
nltk.download("stopwords", quiet=True)

arabic_stopwords = set(stopwords.words("arabic"))


class ArabicTextPreprocessor:

    def __init__(self, remove_stopwords=True, use_stemming=True):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.stemmer = ISRIStemmer()

    def remove_diacritics(self, text):
        """Remove Arabic diacritics and tatweel."""
        if not isinstance(text, str):
            return ""

        arabic_diacritics = re.compile(
            r"[\u064B-\u065F\u0670\u0640]"
        )
        return re.sub(arabic_diacritics, "", text)

    def normalize_text(self, text):
        """Normalize Arabic text and remove punctuation."""
        if not isinstance(text, str):
            return ""

        # Remove diacritics
        text = self.remove_diacritics(text)

        # Keep Arabic letters and whitespace only
        text = re.sub(
            r"[^\u0621-\u063A\u0641-\u064A\s]",
            " ",
            text
        )

        # Normalize Alif forms
        text = re.sub(r"[إأآا]", "ا", text)

        # Normalize Alif Maqsura
        text = re.sub(r"ى", "ي", text)

        # Normalize Hamza forms
        text = re.sub(r"ؤ", "و", text)
        text = re.sub(r"ئ", "ي", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def remove_stop_words(self, text):
        """Remove Arabic stopwords."""
        words = text.split()

        filtered = [
            word for word in words
            if word not in arabic_stopwords
        ]

        return " ".join(filtered)

    def stem_text(self, text):
        """Apply the Arabic ISRI stemmer."""
        words = text.split()

        stemmed = [
            self.stemmer.stem(word)
            for word in words
        ]

        return " ".join(stemmed)

    def process(self, text):
        """Apply the complete preprocessing pipeline."""
        if not isinstance(text, str):
            return ""

        text = self.normalize_text(text)

        if self.remove_stopwords:
            text = self.remove_stop_words(text)

        if self.use_stemming:
            text = self.stem_text(text)

        return text
