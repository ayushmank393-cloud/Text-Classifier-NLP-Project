import nltk
import string
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.metrics import ConfusionMatrix

# ---------------- NLTK Downloads (Run Once) ----------------
resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
for r in resources:
    try:
        nltk.data.find(r)
    except LookupError:
        nltk.download(r)

# ---------------- Dataset ----------------
data = [
    # Sports
    ("The striker scored a hat-trick", "Sports"),
    ("The tennis match went to five sets", "Sports"),
    ("Olympic athletes train for years", "Sports"),
    ("The basketball team won the championship", "Sports"),

    # Tech
    ("Cloud computing enables scalable applications", "Tech"),
    ("Cybersecurity is crucial for online systems", "Tech"),
    ("The app was built using React and Node.js", "Tech"),
    ("Artificial intelligence is transforming healthcare", "Tech"),

    # Politics
    ("The parliament debated the budget proposal", "Politics"),
    ("The senator addressed the media", "Politics"),
    ("A new policy was announced by the ministry", "Politics"),
    ("The opposition criticized the decision", "Politics"),

    # Business
    ("The company reported record profits", "Business"),
    ("Stock markets closed higher today", "Business"),
    ("The startup secured venture capital funding", "Business"),
    ("Inflation affects consumer spending", "Business"),

    # Entertainment
    ("The movie broke all box office records", "Entertainment"),
    ("The actor won an award for best performance", "Entertainment"),
    ("The concert was attended by thousands", "Entertainment"),
    ("A new web series was released online", "Entertainment")
]

# ---------------- Preprocessing ----------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
punctuation = set(string.punctuation)


def preprocess(text: str):
    """Tokenize, remove stopwords, punctuation, and lemmatize"""
    words = word_tokenize(text.lower())
    clean_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and word not in punctuation
    ]
    return clean_words


def extract_features(text: str) -> dict:
    """
    Uses word frequency instead of boolean features
    """
    words = preprocess(text)
    features = {}

    for word in words:
        features[word] = features.get(word, 0) + 1

    return features


# ---------------- Feature Set ----------------
feature_set = [(extract_features(text), label) for text, label in data]

# Shuffle for randomness
random.shuffle(feature_set)

# Train-Test Split (80-20)
split_point = int(0.8 * len(feature_set))
train_set = feature_set[:spl]()
