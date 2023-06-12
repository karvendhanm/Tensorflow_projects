import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Count Vectorizer
corpus = [
    'I like fruits. Fruits like bananas',
    'I love bananas but eat an apple',
    'An apple a day keeps the doctor away'
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

vectorizer.get_feature_names_out()
X.toarray()

# to compare similarity between documents
cosine_similarity(X)

# when an unknown token is encountered for transform operation
# after the fit has happened in CountVectorizer, CountVectorizer
# simply ignores it.
query = vectorizer.transform(['Mini apple and bananas'])
# the word "Mini" has been ignored.
query.toarray()

cosine_similarity(X, query)

# tfidf vectorizer and transformer.
# use_idf = False gives us normalized term frequency.
tfidf_transform = TfidfTransformer(use_idf=False)
X_tf = tfidf_transform.fit_transform(X)
X_tf.toarray()

