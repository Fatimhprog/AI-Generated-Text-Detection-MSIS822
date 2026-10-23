import re
import nltk
from nltk.stem.isri import ISRIStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords
arabic_stopwords = set(stopwords.words('arabic'))

class ArabicTextPreprocessor:

    def __init__(self, remove_stopwords=True, use_stemming=True):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.stemmer = ISRIStemmer()

    
    class ArabicTextPreprocessor:

    def remove_diacritics(self, text):
        arabic_diacritics = re.compile("""
                                     ّ    | 
                                     َ    | 
                                     ً    | 
                                     ُ    | 
                                     ٌ    | 
                                     ِ    | 
                                     ٍ    | 
                                     ْ    | 
                                     ـ     
                                 """, re.VERBOSE)
        return re.sub(arabic_diacritics, '', text)


    def normalize_text(self, text):
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)  
        text = re.sub(r'[إأآا]', 'ا', text)               
        text = re.sub(r'ى', 'ي', text)                  
        text = re.sub(r'ؤ', 'و', text)                  
        text = re.sub(r'ئ', 'ي', text)                  
        text = re.sub(r'\s+', ' ', text).strip()      
        return text

    
    def remove_stop_words(self, text):
        words = text.split()
        filtered = [w for w in words if w not in arabic_stopwords]
        return " ".join(filtered)

 
    def stem_text(self, text):
        words = text.split()
        stemmed = [self.stemmer.stem(w) for w in words]
        return " ".join(stemmed)

    
    def process(self, text):
        if not isinstance(text, str):
            return ""

        text = self.remove_diacritics(text)
        text = self.normalize_text(text)

        if self.remove_stopwords:
            text = self.remove_stop_words(text)

        if self.use_stemming:
            text = self.stem_text(text)

        return text
