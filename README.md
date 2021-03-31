# hydra-fighters-dev
Experiments for our future work

# Theory

Text from documents needs to be represented in a quantifiable form, which is a mathematical object called vector. Vector has a magnitude (value) and a direction. Matrix and tensor also are vectors. Vector space (linear space) is a set of vectors, which may be added together and multiplied ("scaled") by numbers, called scalars. Vector spaces are the subject of linear algebra and are well characterized by their dimension, which, roughly speaking, specifies the number of independent directions in the space.

Similarity function: cosine distance (cosine of the angle between 2 vectors) / Euclidean distance.

3 ways to measure the similarity mathematically:
1. TF-IDF + vectorizing.
2. Embeddings - the vector representations of text where word or sentences with similar meaning or context have similar representations.
    - `word2vec` embeds words into vector space. Word2vec takes a text corpus as input and produce word embeddings as output. There are two main learning algorithms in word2vec: continuous __bag of words__ and continuous __skip gram__. We can train our own embeddings if have enough data and computation available or we can use pre-trained embeddings.

# Development

## Approaches to test models

1. GloVe. I can take the code from tutorial and replace the .txt file from Natasha project
    -> is already a dict with key = word and value = list of factors
    -> для понимания можно сначала это почитать: https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
2. BERT
    - vectorize words + use BERT (package: transformers) (нужно скачать русскую модель и так же использовать)
        - RU BERT from DeepPavlov
        - Slovnet BERT NER — аналог DeepPavlov BERT NER. Reference: https://natasha.github.io/ner/ Source: https://github.com/natasha/slovnet/blob/master/scripts/02_bert_ner/main.ipynb
    - vectorize sentences / paragraphs + use package: sentence-transformers
        - https://www.sbert.net/docs/pretrained_models.html
        - Аналог от DeepPavlov: https://huggingface.co/DeepPavlov/rubert-base-cased-sentence
        - https://huggingface.co/sberbank-ai/sbert_large_nlu_ru
        - English pre-trained model (translate into English via API?): https://tfhub.dev/google/universal-sentence-encoder-large/5
        - Russian again: https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3 --> tutorial: https://d-salvaggio.medium.com/multilingual-universal-sentence-encoder-muse-f8c9cd44f171

Try this: 
- https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3
- https://d-salvaggio.medium.com/multilingual-universal-sentence-encoder-muse-f8c9cd44f171

## Dependencies management
1. Install conda (preferred over pip, because it handles binaries, which might be important for a ML project): https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
2. `>>> source /home/<user>/miniconda3/bin/activate` (or another path to the conda's `activate` file)
3. `>>> conda create --name nlp python=3.8`
4. `>>> conda activate nlp`
5. `>>> conda install --file requirements.txt` (executed inside of your repo)
