import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
texts = ["Good product" for _ in range(50)] + ["Bad quality" for _ in range(50)]
labels = ["good"] * 50 + ["bad"] * 50
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=300)
X = vectorizer.fit_transform(texts)
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))

def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)
