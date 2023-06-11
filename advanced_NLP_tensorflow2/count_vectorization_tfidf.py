from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

query = vectorizer.transform(['apple and bananas'])
query.toarray()

# checking the similarity of the documents wit thw given key words.
cosine_similarity(X, query)

