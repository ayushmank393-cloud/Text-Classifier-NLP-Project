import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

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

stop_words = set(stopwords.words("english"))
punctuation = set(string.punctuation)


def extract_features(text: str) -> dict:
    """
    Converts text into a feature dictionary
    """
    words = word_tokenize(text.lower())
    features = {}

    for word in words:
        if word not in stop_words and word not in punctuation:
            features[word] = True

    return features

feature_set = [(extract_features(text), label) for text, label in data]

train_set = feature_set[:16]
test_set = feature_set[16:]

classifier = NaiveBayesClassifier.train(train_set)

print("Model Accuracy:", accuracy(classifier, test_set))

print("\nMost Informative Features:")
classifier.show_most_informative_features(10)

while True:
    test_text = input("\nEnter a sentence (or type 'exit'): ")
    if test_text.lower() == "exit":
        break

    result = classifier.classify(extract_features(test_text))
    print("Predicted Category:", result)
