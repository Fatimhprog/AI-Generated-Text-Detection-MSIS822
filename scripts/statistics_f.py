import re

class StatisticalFeatures:

    def avg_word_length(self, text):
        words = text.split()
        if len(words) == 0:
            return 0
        return sum(len(w) for w in words) / len(words)

    def avg_sentence_length(self, text):
        sentences = re.split(r"[.!؟]", text)
        num_sentences = len([s for s in sentences if s.strip() != ""])
        words = re.findall(r"\w+", text)
        if num_sentences == 0:
            return 0
        return len(words) / num_sentences

    def ttr(self, text):
        words = text.split()
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)

    def extract_all(self, text):
        return {
            "avg_word_length": self.avg_word_length(text),
            "avg_sentence_length": self.avg_sentence_length(text),
            "ttr": self.ttr(text)
        }
