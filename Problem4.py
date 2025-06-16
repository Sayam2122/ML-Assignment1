import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
reviews = ["Good movie" for _ in range(50)] + ["Bad acting" for _ in range(50)]
sentiments = ["positive"] * 50 + ["negative"] * 50
df = pd.DataFrame({"Review": reviews, "Sentiment": sentiments})
vectorizer = CountVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(df["Review"])
y = df["Sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

def predict_review_sentiment(model, vectorizer, review):
    vec = vectorizer.transform([review])
    return model.predict(vec)[0]
