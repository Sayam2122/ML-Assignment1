import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

tf = []
for doc in corpus:
    words = doc.split()
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    total_words = len(words)
    tf_doc = {}
    for word in word_count:
        tf_doc[word] = word_count[word] / total_words
    tf.append(tf_doc)

df = {}
for doc in corpus:
    words = set(doc.split())
    for word in words:
        df[word] = df.get(word, 0) + 1

idf = {}
N = len(corpus)
for word in df:
    idf[word] = math.log(N / df[word])

manual_tfidf = []
for doc_tf in tf:
    tfidf_doc = {}
    for word in doc_tf:
        tfidf_doc[word] = doc_tf[word] * idf[word]
    manual_tfidf.append(tfidf_doc)

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(corpus)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

print("Manual TF-IDF:")
for doc in manual_tfidf:
    print(doc)

print("\nCountVectorizer (Vocabulary):")
print(count_vectorizer.vocabulary_)
print(count_matrix.toarray())

print("\nTfidfVectorizer (Vocabulary):")
print(tfidf_vectorizer.vocabulary_)
print(tfidf_matrix.toarray())