import pandas as pd
import nltk
import gensim.downloader as api
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return words

def average_vector(words, w2v_model):
    vectors = [w2v_model[word] for word in words if word in w2v_model]
    if len(vectors) == 0:
        return [0]*300
    return sum(vectors) / len(vectors)

w2v_model = api.load("word2vec-google-news-300")

spam_df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
spam_df.columns = ['label', 'message']
spam_df['tokens'] = spam_df['message'].apply(lambda x: preprocess_text(x))
spam_df['vector'] = spam_df['tokens'].apply(lambda x: average_vector(x, w2v_model))
X = list(spam_df['vector'])
y = spam_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Spam Accuracy:", accuracy_score(y_test, y_pred))

def predict_message_class(model, w2v_model, message):
    words = preprocess_text(message)
    vector = average_vector(words, w2v_model)
    return model.predict([vector])[0]
