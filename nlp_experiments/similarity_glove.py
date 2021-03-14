'''
Reference: https://towardsdatascience.com/calculating-document-similarities-using-bert-and-other-models-b2c1a29c9630

GloVe: unsupervised learning algorithm to produce vector repr. of a word.
There are pre-trained embeddings - same as with Word2Vec.

Difference to word2vec:
- word2vec is a "predictive" model, and GloVe - "count-based"
- experiments on a big dataset are faster with GloVe.

A GloVe file (.txt or tsv) looks like this:
times -0.39921 0.63609 0.3401 -0.14305 -0.11966 -0.4832 -0.87891 -0.19646 -0.42128 -0.23169 -0.50591 -0.093337 -0.30098 -0.14887 1.2242 -0.11392 -0.59522 -0.48003 -0.98778 0.033121 0.17605 0.2695 0.96337 0.25842 0.44454 -1.4805 -0.47694 0.040743 0.15592 0.23329 3.3342 -0.1008 0.85134 -0.29978 -0.14765 -0.37207 0.093149 0.30571 -0.036654 0.12316 0.012305 0.43128 0.4405 0.58298 -0.39951 0.21648 -0.13706 -0.01493 -0.19099 0.24173
took 0.10581 -0.12607 -0.15304 -0.11204 0.17447 0.11011 -0.87909 0.5273 -0.50619 -0.41388 -0.056028 -0.17814 -1.0869 -0.25415 0.43534 -0.34265 -0.27734 -0.35973 -1.1806 -0.15128 0.24127 0.50711 0.17117 -0.47705 -0.023802 -1.7512 0.099813 -0.43687 -0.19324 -0.037552 2.8149 0.41473 -0.48028 -0.014933 0.3202 0.085309 0.19205 0.53356 0.018675 0.077745 -0.25481 0.24782 -0.26002 -0.2175 0.043965 -0.090949 -0.24377 -0.58687 -0.048828 -0.38006
'''

import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

modus = "RUSSIAN_TRIAL"

if modus == "ENGLISH_TUTORIAL":
    # Sample corpus
    documents = [
        'Machine learning is the study of computer algorithms that improve automatically through experience.\
    Machine learning algorithms build a mathematical model based on sample data, known as training data.\
    The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
    where no fully satisfactory algorithm is available.',
        'Machine learning is closely related to computational statistics, which focuses on making predictions using computers.\
    The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.',
        'Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so. \
    It involves computers learning from data provided so that they carry out certain tasks.',
        'Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the "signal"\
    or "feedback" available to the learning system: Supervised, Unsupervised and Reinforcement',
        'Software engineering is the systematic application of engineering approaches to the development of software.\
    Software engineering is a computing discipline.',
        'A software engineer creates programs based on logic for the computer to execute. A software engineer has to be more concerned\
    about the correctness of the program in all the cases. Meanwhile, a data scientist is comfortable with uncertainty and variability.\
    Developing a machine learning application is more iterative and explorative process than software engineering.'
    ]

    model_file = '../glove.6B.100d.txt'

    # reading Glove word embeddings into a dictionary with "word" as key and values as word vectors
    embeddings_index = dict()

    language = 'english'

    # TODO: here to download
    # we just trasform tsv into dict
    with open(model_file) as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

elif modus == "RUSSIAN_TRIAL":
    from navec import Navec
    path = '../navec_news_v1_1B_250K_300d_100q.tar'
    embeddings_index = Navec.load(path)

    documents = [
        # '«На самом деле бело-красно-белый флаг лишь для прозападной оппозиции является «национальным». Вся история этого флага (с 1918 года) — это история предательства, коллаборационизма и других преступлений перед народом Белоруссии. Из наилучших или иных побуждений — вопрос дискуссионный, очевидный факт — именно под этим флагом. Одного лишь соучастия белорусских националистов в геноциде своего (?) народа во время Великой Отечественной войны достаточно, чтобы поставить крест на этой символике. Если уж свастику запретили по понятным причинам, то по бело-красно-белому флагу («БЧБ-символике») такое же решение было бы оправданным».',
        '«На самом деле бело-красно-белый флаг лишь для прозападной оппозиции является «национальным». Вся история этого флага (с 1918 года) — это история предательства, коллаборационизма и других преступлений перед народом Белоруссии. Из наилучших или иных побуждений — вопрос дискуссионный, очевидный факт — именно под этим флагом.',
        'БЧБ-флаг – символ геноцида белорусского народа',
        'БЧБ-флаг – символ славы белорусского народа, который осуществлял уничтожение других народов',
        'Запад пытался захватить власть в Беларуси с помощью мятежников-русофобов',
        'Русофобия и вытеснение русского языка',
        'Литва торгует русофобией и белорусофобией'
    ]
    language = 'russian'
else:
    raise Exception("non-existing modus")

documents_df=pd.DataFrame(documents, columns=['documents'])

# removing special characters and stop words from the text
stop_words_l=stopwords.words(language)
print(stop_words_l)
# here a function for this: https://www.kaggle.com/alxmamaev/how-to-easy-preprocess-russian-text
documents_df['documents_cleaned'] = documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^А-я]', ' ', w).lower() for w in x.split() if re.sub(r'[^А-я]',' ',w).lower() not in stop_words_l))

'''
from pymystem3 import Mystem
from string import punctuation

def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    return text
'''

tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents_df.documents_cleaned)
tokenized_documents = tokenizer.texts_to_sequences(documents_df.documents_cleaned)
# TODO: what is data type of tokenized_documents? List of lists?
tokenized_paded_documents = pad_sequences(tokenized_documents, padding='post')
vocab_size=len(tokenizer.word_index)+1
print(f"tokenized_paded_documents: {tokenized_paded_documents}")  # creates arrays with IDs instead of words (words are coded)

# creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index.
embedding_matrix = np.zeros((vocab_size, 300))  # we create empty vector of size = number of words
# TODO: 300 - to a variable; can be dynamically inspected?

print(f"tokenizer.word_index: {tokenizer.word_index}")  # shows mapping of words to IDs
for word, i in tokenizer.word_index.items():  # TODO: what is i? --> id, like enumerate()
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector  # list of lists, where index of list - ID of words, and contents - embedding for this words

print(f"embedding_matrix: {embedding_matrix}")

tfidfvectoriser=TfidfVectorizer()
tfidfvectoriser.fit(documents_df.documents_cleaned)
tfidf_vectors=tfidfvectoriser.transform(documents_df.documents_cleaned)
tfidf_vectors=tfidf_vectors.toarray()
print(f"tfidf_vectors: {tfidf_vectors}, type: {type(tfidf_vectors)}")
words = tfidfvectoriser.get_feature_names()
print(f"words: {words}")

# calculating average of word vectors of a document weighted by tf-idf
document_embeddings = np.zeros((len(tokenized_paded_documents), 300))
print(f"document_embeddings: {document_embeddings}")

# instead of creating document-word embeddings, directly creating document embeddings
for i in range(documents_df.shape[0]):
    for j in range(len(words)):
        print("############################# checking here ############################")
        print(tokenizer.word_index[words[j]])
        print(tfidf_vectors[i][j])
        # Here we add to each row (= each document from the corpus)
        # 
        document_embeddings[i] += embedding_matrix[
            tokenizer.word_index[words[j]]] * tfidf_vectors[i][j]

document_embeddings=document_embeddings/np.sum(tfidf_vectors,axis=1).reshape(-1,1)

pairwise_similarities=cosine_similarity(document_embeddings)
pairwise_differences=euclidean_distances(document_embeddings)


def most_similar(doc_id, similarity_matrix, matrix):
    print(f'Document: {documents_df.iloc[doc_id]["documents"]}')
    print('\n')
    print('Similar Documents:')
    if matrix == 'Cosine Similarity':
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix == 'Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix == doc_id:
            continue
        print('\n')
        print(f'Document: {documents_df.iloc[ix]["documents"]}')
        print(f'{matrix} : {similarity_matrix[doc_id][ix]}')


most_similar(0,pairwise_similarities,'Cosine Similarity')
most_similar(0,pairwise_differences,'Euclidean Distance')
