from sentence_transformers import SentenceTransformer, util

from test_data import test_data


# model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')  # too big size of model
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')  # not so exact
# model = SentenceTransformer('stsb-xlm-r-multilingual')  # too big size of model

# Two lists of sentences
input_text = 'БЧБ-флаг – символ геноцида белорусского народа. Это нацистский, фашистский флаг.'

narratives = test_data

#Compute embedding for both lists
embeddings1 = model.encode([input_text], convert_to_tensor=True)
embeddings2 = model.encode(narratives, convert_to_tensor=True)
print(embeddings2)

#Compute cosine-similarits
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(narratives)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(input_text, narratives[i], cosine_scores[0][i]))
