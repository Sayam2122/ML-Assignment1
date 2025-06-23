import pandas as pd
import re
import nltk
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

def average_vector(words, w2v_model):
    vectors = [w2v_model[word] for word in words if word in w2v_model]
    if len(vectors) == 0:
        return [0]*300
    return sum(vectors) / len(vectors)

w2v_model = api.load("word2vec-google-news-300")

df = pd.read_csv("Tweets.csv")[['airline_sentiment', 'text']]
df['tokens'] = df['text'].apply(lambda x: preprocess_text(x))
df['vector'] = df['tokens'].apply(lambda x: average_vector(x, w2v_model))
X = list(df['vector'])
y = df['airline_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Sentiment Accuracy:", accuracy_score(y_test, y_pred))

def predict_tweet_sentiment(model, w2v_model, tweet):
    words = preprocess_text(tweet)
    vector = average_vector(words, w2v_model)
    return model.predict([vector])[0]
