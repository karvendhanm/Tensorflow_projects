from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
model_w2v = api.load('word2vec-google-news-300')

model_w2v.most_similar('cookies', topn=10)
model_w2v.doesnt_match(["Washington", 'Ottawa', 'Turkey', 'Tokyo'])

king = model_w2v['king']
man = model_w2v['man']
women = model_w2v['woman']

queen = king - man + women
model_w2v.similar_by_vector(queen)