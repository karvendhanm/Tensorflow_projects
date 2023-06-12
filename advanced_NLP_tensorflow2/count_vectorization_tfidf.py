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

# tfidf transformer.
# tfidf transformer needs the output of CountVectorizer as input or count matrix as input.
# use_idf = False gives us normalized term frequency.
tfidf_transform_tf = TfidfTransformer(use_idf=False)
X_tf = tfidf_transform_tf.fit_transform(X)
X_tf.toarray()

tf_idf_transform_idf = TfidfTransformer(smooth_idf=False)
X_idf = tf_idf_transform_idf.fit_transform(X)
X_idf.toarray()

df_tfidf_transform = pd.DataFrame(X_idf.toarray(), columns=vectorizer.get_feature_names_out())

# tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer()
vec = tfidf_vectorizer.fit_transform(corpus)
vec.toarray()




