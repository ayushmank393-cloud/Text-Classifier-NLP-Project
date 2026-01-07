import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

data = [
    ("The team won the cricket match", "Sports"),
    ("Football players are training hard", "Sports"),
    ("The new smartphone uses AI technology", "Tech"),
    ("Python is used in software development", "Tech"),
    ("The government passed a new law", "Politics"),
    ("Elections will be held next month", "Politics")
]

stop_words = set(stopwords.words("english"))


def extract_features(text):
    words = word_tokenize(text.lower())
    features = {}
    for word in words:
        if word not in stop_words:
            features[word] = True
    return features


feature_set = [(extract_features(text), label) for text, label in data]


classifier = NaiveBayesClassifier.train(feature_set)


test_text = input("Enter a sentence: ")
result = classifier.classify(extract_features(test_text))

print("Category:", result)
