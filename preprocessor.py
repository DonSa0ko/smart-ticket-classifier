"""
preprocessor.py
---------------
Shared text preprocessing transformer used by both train.py and classifier.py.

Defined as a standalone module so that pickle can locate the TextPreprocessor
class when loading saved models. If this class were defined inside train.py,
the models would fail to load in any other context.
"""

import re
import string

from sklearn.base import BaseEstimator, TransformerMixin

# Common English words that carry no classification value
STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "my", "i", "me", "we", "be",
    "was", "are", "has", "have", "had", "do", "does", "did", "with",
    "this", "that", "from", "by", "as", "up", "out", "so", "if",
    "its", "than", "then", "when", "will", "can", "just", "been",
    "also", "very", "too", "all", "any", "no", "after", "before",
    "some", "our", "their", "your", "his", "her", "they", "them"
}


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that cleans raw ticket text.

    Steps applied to each ticket:
        1. Lowercase the text
        2. Remove punctuation
        3. Remove numeric tokens (e.g. port numbers, IP fragments)
        4. Remove common stopwords
        5. Strip extra whitespace
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._clean(text) for text in X]

    def _clean(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b\d+\b", "", text)
        tokens = [w for w in text.split() if w not in STOPWORDS]
        return " ".join(tokens)