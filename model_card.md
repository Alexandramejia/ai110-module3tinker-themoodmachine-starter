# Model Card: Mood Machine

This model card covers two versions of the Mood Machine classifier:

1. A **rule-based model** in `mood_analyzer.py`
2. A **machine learning model** in `ml_experiments.py` using scikit-learn

---

## 1. Model Overview

**Model type:**  
I built and primarily used the rule-based model, but also implemented the ML model for comparison.

**Intended purpose:**  
Classify short, informal text posts into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`. Think social media captions, text messages, quick status updates — not full essays.

**How it works (brief):**  
The rule-based version scans each post for known positive and negative words, flips word scores when a negation word like "not" or "never" appears right before them, and then decides a label based on the final numeric score. If a post has both positive and negative words, it gets labeled `mixed` regardless of the score total.

The ML version converts posts into a bag-of-words vector using `CountVectorizer`, then trains a logistic regression classifier on the labeled examples in `SAMPLE_POSTS`. It learns which words tend to appear in each label category from the training data — no hand-coded rules.

---

## 2. Data

**Dataset description:**  
The dataset started with 6 posts and I added 8 more, bringing the total to 14. They're all short, informal sentences in the style of social media posts or texts.

**Labeling process:**  
I labeled each post based on the overall feeling it expresses, not just the literal words. That meant thinking about context — "mad happy" is positive because "mad" is slang for "very", and "I absolutely love getting stuck in traffic 🙃" is negative because the upside-down face is a sarcasm signal. A few posts genuinely could go either way; for those I picked the label that felt more accurate and left a comment explaining the ambiguity.

Some posts were genuinely hard to call:
- `"This is fine"` — could be neutral acceptance or sarcastic frustration. I labeled it `neutral` but a lot of people would read it as sarcasm.
- `"stressed but blessed fr"` — both feelings are real and present, so `mixed` felt right, but you could argue it trends positive.
- `"vibing so hard right now, life is good 💀"` — the skull emoji in Gen Z slang means dying of laughter (positive), not death (negative). Labeled `positive`, but someone unfamiliar with that usage would likely disagree.

**Important characteristics of the dataset:**

- Contains Gen Z slang: `lowkey`, `no cap`, `bussin`, `goated`, `lit`, `vibing`, `fr`
- Uses emojis as sentiment signals: 😂, 💀, 🙃
- Includes sarcasm that flips the literal meaning
- Has mixed-feeling posts where both positive and negative apply
- Short and informal — no full sentences in some cases

**Possible issues:**  
The dataset is very small (14 examples). Most posts lean toward English-speaking Gen Z slang, so anyone writing in a different dialect or style is not well-represented. There's also a class imbalance: more `mixed` and `positive` examples than `neutral` or `negative` ones.

---

## 3. How the Rule-Based Model Works

**Scoring rules:**

- Each word in the post is checked against `POSITIVE_WORDS` and `NEGATIVE_WORDS`
- Positive words add `+1` to the score; negative words subtract `1`
- If the word immediately before a scored word is `not`, `never`, or `no`, the score effect is flipped (so "not happy" scores `-1` instead of `+1`)
- If a post contains at least one positive AND one negative word, it returns `mixed` regardless of the final score
- If neither condition applies: score > 0 → `positive`, score < 0 → `negative`, score == 0 → `neutral`

Preprocessing also does some work before scoring: it swaps ASCII emoticons like `:)` and `:(` into words (`happy`, `sad`), maps common Unicode emojis to sentiment words, normalizes repeated characters (`soooo` → `soo`), and strips punctuation.

**Strengths:**  
Works well on clear, direct language. Posts like "I love this class so much" or "Today was a terrible day" are easy calls. The negation logic helps catch "I am not happy about this" correctly. Emoji handling actually works well for the most common cases (😂 = happy, 😭 = sad, 🙃 = sad/sarcasm signal).

**Weaknesses:**  
Sarcasm is the biggest failure. When someone writes "I absolutely love getting stuck in traffic 🙃", every word except the emoji reads as positive, so the model sees both "love" and the emoji-derived "sad" word and calls it `mixed` — when it's clearly negative. The model has no understanding of sentence-level irony.

Negation also only looks back one word, so "I am not really that happy today" would miss the negation of "happy" because "really" sits between "not" and "happy."

---

## 4. How the ML Model Works

**Features used:**  
Bag of words using `CountVectorizer`. Each post is turned into a vector where each dimension is a word from the training vocabulary, and the value is how many times that word appears.

**Training data:**  
The model trains on `SAMPLE_POSTS` with `TRUE_LABELS` from `dataset.py` — the same 14 labeled examples used to evaluate the rule-based model.

**Training behavior:**  
Because the model trains and evaluates on the same 14 examples, accuracy will look high — it's essentially memorizing the training set. The model is very sensitive to the labels assigned. Changing a label on even one or two posts can visibly shift what the model predicts on related inputs, since there are so few examples per class. Adding more labeled data (especially for `neutral`) would help it generalize.

**Strengths and weaknesses:**  
The ML model can pick up on word patterns it wasn't explicitly told to look for, which is useful. But with only 14 training examples, it's almost certainly overfitting — it's learned the specific wording of these posts rather than the concept of "negative" language. It also can't handle negation unless negation phrases like "not happy" happened to appear in its training data with a negative label, which is fragile. It would likely make very different predictions on new posts that use any unfamiliar words.

---

## 5. Evaluation

**How I evaluated:**  
Both models were run against the full `SAMPLE_POSTS` list with `TRUE_LABELS` as the ground truth.

**Rule-based model accuracy: 86% (12/14 correct)**

**Examples of correct predictions:**

- `"I love this class so much"` → `positive` ✓  
  "Love" is a strong positive word and there are no negative words present. Straightforward.

- `"I am not happy about this"` → `negative` ✓  
  The negation rule fired: "not" precedes "happy", so instead of scoring +1 it scored -1. This is the intended behavior.

- `"im lowkey drained from work but still grateful"` → `mixed` ✓  
  "Drained" is negative and "grateful" is positive — the mixed-label logic caught both.

**Examples of incorrect predictions:**

- `"I absolutely love getting stuck in traffic 🙃"` → predicted `mixed`, true `negative`  
  The model sees "love" (positive) and "sad" (from the 🙃 emoji conversion) and calls it mixed. It can't recognize that the whole sentence is sarcastic. The word "love" is genuinely there and genuinely positive to the rule-checker — there's nothing in the scoring logic that accounts for "love + bad situation = sarcasm."

- `"not sad just tired of everything"` → predicted `neutral`, true `negative`  
  "Sad" gets negated by "not", reducing the score. "Tired" is in the negative word list, so that should drag the score down — but after negation flips "sad" to +1 and "tired" gives -1, the score lands at 0, resulting in `neutral`. The intended label is `negative` because the post is clearly expressing exhaustion and frustration. The problem is that negating "sad" overcompensates — just because someone isn't sad doesn't mean they're doing okay.

---

## 6. Limitations

- **Dataset is tiny.** 14 posts is not enough to be confident in any pattern. A single mislabeled example can measurably change ML model accuracy.

- **Sarcasm is invisible to the rule-based model.** There's no mechanism to detect when positive words are being used ironically. This is a known hard problem even for large language models.

- **Negation only looks one word back.** "I'm not really that happy" would miss the negation because "really" breaks the `not + word` adjacency assumption.

- **Slang coverage is partial.** The word lists include some Gen Z slang (`bussin`, `goated`, `fire`, `lit`) but these words change meaning fast and the list would need constant updating to stay current.

- **No generalization beyond the word list.** If someone writes "Today wrecked me" using a word not in `NEGATIVE_WORDS`, the model has no way to score it — it would return `neutral` by default.

- **The "mixed" label can mask real signal.** Anytime both a positive and negative word appear, the post is labeled mixed regardless of proportion. A post with 5 negative words and 1 positive word gets the same label as a genuinely ambivalent post.

---

## 7. Ethical Considerations

**Misclassifying distress:**  
If this model were used in a real app to flag mood, a post like "not sad just tired of everything" getting labeled `neutral` instead of `negative` could mean someone expressing exhaustion goes unnoticed. In any context where mood detection is being used to identify users who might need support, false neutrals are a real problem.

**Sarcasm misclassification:**  
"I absolutely love getting stuck in traffic" being called `mixed` instead of `negative` seems harmless, but scale that up: if this model is being used to analyze social media sentiment about a product or service, sarcastic negative feedback would be systematically undercounted.

**Language bias:**  
The dataset is centered on informal American English and Gen Z internet slang. The word lists and examples assume a specific cultural context. Someone writing in a different dialect, a different emotional register, or mixing in another language would likely get worse predictions. The model wasn't designed with that in mind, and there's nothing in the architecture that would help it adapt.

**Privacy:**  
Mood classification means processing personal, often emotional content. Even for a class project, it's worth noting that running this kind of analysis on real posts without user consent raises privacy questions — especially if mood data were ever stored or used to make decisions about people.

---

## 8. Ideas for Improvement

- **More data.** Even 100 labeled examples would help the ML model generalize. For the rule-based model, more coverage in the word lists (especially for neutrals and mixed expressions) would reduce edge case failures.

- **Better negation scope.** Instead of just checking the immediately preceding word, look back up to 3 words for negation terms. This would handle "not really that happy" and similar constructions.

- **TF-IDF instead of raw counts.** Replacing `CountVectorizer` with `TfidfVectorizer` would down-weight common words that appear across all posts (like "I", "the") and give more weight to distinctive terms.

- **Separate test set.** Right now both models evaluate on the same data they were built from. Adding a held-out test set — even just 5-10 posts the model never saw — would give a more honest picture of real performance.

- **Sarcasm detection heuristics.** One rough approach: if a post contains a known sarcasm emoji (🙃, 🙄) or a phrase like "I absolutely love" followed by an objectively unpleasant noun, penalize the positive score. Not perfect, but better than nothing.

- **Transformer model.** A small pre-trained model like DistilBERT, fine-tuned on a few hundred labeled examples, would handle sarcasm, context, and negation far better than either approach here. That's a bigger lift, but it would fix most of the limitations described above.
