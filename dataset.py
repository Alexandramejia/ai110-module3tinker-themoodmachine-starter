"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # new
    "lowkey",
    "blessed",
    "grateful",
    "proud",
    "vibing",
    "thrilled",
    "lit",
    "hopeful",
    "close",
    "sick",
    "wicked",
    "fire",
    "goated",
    "bussin",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # new
    "miserable",
    "frustrated",
    "exhausted",
    "drained",
    "depressed",
    "rough",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",   # tired vs hopeful — which wins?
    "This is fine",                         # neutral or sarcasm?
    "So excited for the weekend",
    "I am not happy about this",            # "not" flips it — does the model catch that?
    # new
    "im lowkey drained from work but still grateful",
    "im mad happy rn no cap 😂",            # "mad" = slang for "very" here, not angry
    "I absolutely love getting stuck in traffic 🙃",  # sarcasm — words say positive, meaning is negative
    "honestly exhausted but proud of how far I've come",
    "this week has been rough but the weekend is so close",
    "vibing so hard right now, life is good 💀",  # 💀 used positively in gen z slang
    "not sad just tired of everything",     # "sad" negated but still negative
    "stressed but blessed fr",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful" — could argue either way
    "neutral",   # "This is fine" — neutral or sarcasm?
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this" — does the model catch "not"?
    # new
    "mixed",     # "im lowkey drained from work but still grateful"
    "positive",  # "im mad happy rn no cap 😂" — "mad" as slang, watch rule-based get this wrong
    "negative",  # "I absolutely love getting stuck in traffic 🙃" — sarcasm flips it
    "mixed",     # "honestly exhausted but proud of how far I've come"
    "mixed",     # "this week has been rough but the weekend is so close"
    "positive",  # "vibing so hard right now, life is good 💀" — would you label differently bc of 💀?
    "negative",  # "not sad just tired of everything" — "sad" negated but still negative
    "mixed",     # "stressed but blessed fr"
]

#                   !!!! could have also added entries using .append() like this: !!!!
# SAMPLE_POSTS.append("im lowkey drained from work but still grateful")
# TRUE_LABELS.append("mixed")

#                   !!! Part 1: Once you've finished adding words and posts above, run: !!!
#   python main.py
# This confirms the program doesn't crash with your additions.
# Predictions will show None until score_text and predict_label
# in mood_analyzer.py are implemented — that's expected for now.


# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
