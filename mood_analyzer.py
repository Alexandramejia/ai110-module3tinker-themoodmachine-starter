# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()

        # Swap emoticons/emojis for plain words BEFORE removing punctuation,
        # so the scorer can still read them as positive/negative signals.
        # Replace ASCII emoticons with sentiment words before punctuation is stripped
        ascii_emoticons = {
            ':-)': 'happy', ':)': 'happy', '=)': 'happy',
            ':-(': 'sad',   ':(': 'sad',   ":'(": 'sad',
            ':-d': 'happy', ':d': 'happy',
            ';-)': 'happy', ';)': 'happy',
            ':-/': 'uncertain', ':/': 'uncertain',
        }
        for emoticon, word in ascii_emoticons.items():
            cleaned = cleaned.replace(emoticon, f' {word} ')

        # Replace common Unicode emojis with sentiment words
        unicode_emojis = {
            '😂': 'happy', '😊': 'happy', '😍': 'happy', '🥰': 'happy',
            '😀': 'happy', '😁': 'happy', '🎉': 'happy', '❤️': 'happy',
            '😢': 'sad',   '😭': 'sad',   '😞': 'sad',   '💔': 'sad',
            '🥲': 'sad',   '😩': 'sad',
            '😠': 'angry', '😡': 'angry', '🤬': 'angry', '😤': 'angry',
        }
        for emoji, word in unicode_emojis.items():
            cleaned = cleaned.replace(emoji, f' {word} ')

        # Drop any remaining non-ASCII characters (other emojis, symbols)
        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)

        # Normalize repeated characters: "sooooo" -> "soo" (cap at 2)
        cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)

        # Remove everything that isn't a letter or whitespace
        cleaned = re.sub(r'[^a-z\s]', '', cleaned)

        tokens = [t for t in cleaned.split() if t]
        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        tokens = self.preprocess(text)
        score = 0

        # Loop through each word in the sentence and check if it's in our word lists.
        # If the word right before it is "not"/"never"/"no", flip the score effect
        # (e.g. "not happy" scores -1 instead of +1, "not bad" scores +1 instead of -1).
        for i, token in enumerate(tokens):
            negated = i > 0 and tokens[i - 1] in {'not', 'never', 'no'}

            if token in self.positive_words:
                score += -1 if negated else 1
            elif token in self.negative_words:
                score += 1 if negated else -1

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score = self.score_text(text)
        tokens = self.preprocess(text)

        # Check whether the sentence contains both positive and negative words
        has_positive = any(t in self.positive_words for t in tokens)
        has_negative = any(t in self.negative_words for t in tokens)

        # If both sides appear, call it mixed regardless of the final score
        if has_positive and has_negative:
            return "mixed"
        elif score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        # We call preprocess here to get the word list so we can collect which words
        # were positive/negative hits and show them in the output.
        # score_text and predict_label also call preprocess internally but only
        # return a number/label — they don't share the word list back.
        tokens = self.preprocess(text)
        positive_hits = [t for t in tokens if t in self.positive_words]
        negative_hits = [t for t in tokens if t in self.negative_words]

        score = self.score_text(text)
        label = self.predict_label(text)

        return (
            f"Label: {label} | Score: {score} | "
            f"positive words: {positive_hits} | negative words: {negative_hits}"
        )
