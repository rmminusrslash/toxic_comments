import logging
import pickle
import re

import numpy as np
import pandas as pd
import tqdm
from keras.preprocessing import text, sequence
from unidecode import unidecode

from spell_correct import SpellCorrect

logging.basicConfig()
logger = logging.getLogger("preprocessing")
logger.setLevel(logging.INFO)


def encoded_data(protopying=True, from_pickle=False):

    if from_pickle:
        x_train_encoded_padded_sequences=pickle.load(open("data/x_train_encoded_padded_sequences", "rb"))
        y_train=pickle.load(open("data/y_train", "rb"))
        x_test_encoded_padded_sequences=pickle.load(open("data/x_test_encoded_padded_sequences", "rb"))
        word_index=pickle.load(open("data/word_index", "rb"))
        return x_train_encoded_padded_sequences, x_test_encoded_padded_sequences, y_train, word_index


    logger.info("Cleaning train and test data")
    train = pd.read_csv("data/train.csv", nrows= 1000 if protopying else None)
    test = pd.read_csv("data/test.csv", nrows= 1000 if protopying else None)

    # 2.  remove non-ascii
    special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]', re.IGNORECASE)

    def clean_text(x):
        x_ascii = unidecode(x)
        x_clean = special_character_removal.sub('', x_ascii)
        return x_clean

    train['clean_text'] = train['comment_text'].apply(lambda x: clean_text(str(x)))
    test['clean_text'] = test['comment_text'].apply(lambda x: clean_text(str(x)))

    X_train = train['clean_text'].fillna("something").values
    y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    X_test = test['clean_text'].fillna("something").values

    logger.info("Encoding and padding sequences")
    tokenizer = text.Tokenizer(num_words=None,oov_token="oov")
    tokenizer.fit_on_texts(list(X_train))

    logger.info("Most common words: %s" % tokenizer.word_index)

    maxlen = 900
    X_train_sequence = tokenizer.texts_to_sequences(X_train)
    X_test_sequence = tokenizer.texts_to_sequences(X_test)

    x_train_encoded_padded_sequences = sequence.pad_sequences(X_train_sequence, maxlen=maxlen)
    x_test_encoded_padded_sequences = sequence.pad_sequences(X_test_sequence, maxlen=maxlen)

    pickle.dump(x_train_encoded_padded_sequences, open("data/x_train_encoded_padded_sequences", "wb"))
    pickle.dump(x_test_encoded_padded_sequences, open("data/x_train_encoded_padded_sequences", "wb"))
    pickle.dump(y_train, open("data/y_train", "wb"))
    pickle.dump(tokenizer.word_index, open("data/word_index", "wb"))

    return x_train_encoded_padded_sequences, x_test_encoded_padded_sequences, y_train, tokenizer.word_index


def read_embeddings(from_picked=True):
    logger.info("Reading embeddings")

    if from_picked:
        embeddings_index_ft = pickle.load(open("data/embeddings_index_ft", "rb"))
        embeddings_index_tw = pickle.load(open("data/embeddings_index_tw", "rb"))
    else:
        EMBEDDING_FILE_FASTTEXT = "data/crawl-300d-2M.vec"
        EMBEDDING_FILE_TWITTER = "data/glove.twitter.27B.200d.txt"

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embeddings_index_ft = dict(
            get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE_FASTTEXT, encoding='utf-8'))
        embeddings_index_tw = dict(
            get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE_TWITTER, encoding='utf-8'))
    return embeddings_index_ft, embeddings_index_tw


def index_to_embedding(fast_text, glove, word_index, from_picked=True):
    if from_picked:
        return pickle.load(open("data/matrix", "rb"))

    else:
        spell_checker = SpellCorrect()
        nb_words = len(word_index) + 1
        embedding_matrix = np.zeros((nb_words, 501))

        something_tw = glove.get("something")
        something_ft = fast_text.get("something")

        something = np.zeros((501,))
        something[:300, ] = something_ft
        something[300:500, ] = something_tw
        something[500,] = 0

        def all_caps(word):
            return len(word) > 1 and word.isupper()

        def embed_word(embedding_matrix, i, word):
            embedding_vector_ft = fast_text.get(word)
            if embedding_vector_ft is not None:
                if all_caps(word):
                    last_value = np.array([1])
                else:
                    last_value = np.array([0])
                embedding_matrix[i, :300] = embedding_vector_ft
                embedding_matrix[i, 500] = last_value
                embedding_vector_tw = glove.get(word)
                if embedding_vector_tw is not None:
                    embedding_matrix[i, 300:500] = embedding_vector_tw

        # Fasttext vector is used by itself if there is no glove vector but not the other way around.
        not_found = 0
        f = 0
        for word, i in tqdm.tqdm(word_index.items()):
            if word in fast_text:
                embed_word(embedding_matrix, i, word)
            else:
                # change to > 20 for better score.
                if len(word) > 0:
                    # print(word)

                    embedding_matrix[i] = something
                    f += 1
                else:
                    word2 = spell_checker.correction(word)
                    if word2 in fast_text:
                        embed_word(embedding_matrix, i, word2)
                    else:
                        word2 = spell_checker.correction(spell_checker.singlify(word))
                        if word2 in fast_text:
                            embed_word(embedding_matrix, i, word2)
                        else:
                            embedding_matrix[i] = something
                            not_found += 1
        print(f, not_found, len(word_index),embedding_matrix.shape)
        #pickle.dump(embedding_matrix, open("data/matrix", "wb"))
        return embedding_matrix


def create_embedding_matrix(word_index, from_picked=True):
    fast_text, glove = read_embeddings(from_picked)
    matrix = index_to_embedding(fast_text, glove, word_index, from_picked)
    return matrix


def preprocess(from_picked=True):
    x_train, x_test, y_train, word_index = encoded_data()
    matrix = create_embedding_matrix(word_index, from_picked)
    return x_train, x_test, y_train, matrix


if __name__ == '__main__':
    preprocess(False)
