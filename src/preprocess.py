import logging
import pickle
import re

import numpy as np
import pandas as pd
import tqdm
from keras.preprocessing import text, sequence
from unidecode import unidecode

from spell_correct import SpellCorrect

logger = logging.getLogger("preprocessing")


def encode_sequences(protopying=False, from_pickle=False, with_spell=False):
    if from_pickle:
        x_train_encoded_padded_sequences = pickle.load(
            open(f"data/x_train_encoded_padded_sequences_spell{with_spell}",
                 "rb"))
        y_train = pickle.load(open("data/y_train", "rb"))
        x_test_encoded_padded_sequences = pickle.load(
            open(f"data/x_test_encoded_padded_sequences_spell{with_spell}",
                 "rb"))
        word_index = pickle.load(
            open(f"data/word_index_spell_{with_spell}", "rb"))
        return x_train_encoded_padded_sequences, x_test_encoded_padded_sequences, y_train, word_index

    logger.info("Cleaning train and test data")
    train = pd.read_csv("data/train.csv", nrows=10000 if protopying else None)
    test = pd.read_csv("data/test.csv", nrows=10000 if protopying else None)

    REGEX_PUNCTUATION = re.compile(
        r"[!\"#$%&()*+,\-.\/:;<=>?@\[\]\\\\\^\_`{|}~\t\n\xa0'\n\r\t']")

    if with_spell:
        spell_checker = SpellCorrect()

    def to_asci_clean_punctuation(x):
        x_ascii = unidecode(x)
        x_clean = REGEX_PUNCTUATION.sub(' ', x_ascii)
        if with_spell:
            x_clean = " ".join([
                spell_checker.correction(word) for word in x_clean.split()
                if word != ""
            ])
        return x_clean

    train['clean_text'] = train['comment_text'].apply(
        lambda x: to_asci_clean_punctuation(str(x)))
    test['clean_text'] = test['comment_text'].apply(
        lambda x: to_asci_clean_punctuation(str(x)))

    if with_spell:
        print(spell_checker.corrected, spell_checker.uncorrected)

    X_train = train['clean_text'].fillna("something").values
    y_train = train[[
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
    ]].values
    X_test = test['clean_text'].fillna("something").values

    logger.info("Encoding and padding sequences")
    tokenizer = text.Tokenizer(num_words=None, oov_token="oov")
    tokenizer.fit_on_texts(list(X_train))

    logger.info("Most common words: %s" %
                list(tokenizer.word_index.items())[:10])

    maxlen = 900
    X_train_sequence = tokenizer.texts_to_sequences(X_train)
    X_test_sequence = tokenizer.texts_to_sequences(X_test)

    x_train_encoded_padded_sequences = sequence.pad_sequences(X_train_sequence,
                                                              maxlen=maxlen)
    x_test_encoded_padded_sequences = sequence.pad_sequences(X_test_sequence,
                                                             maxlen=maxlen)

    pickle.dump(
        x_train_encoded_padded_sequences,
        open(f"data/x_train_encoded_padded_sequences_spell{with_spell}", "wb"))
    pickle.dump(
        x_test_encoded_padded_sequences,
        open(f"data/x_test_encoded_padded_sequences_spell{with_spell}", "wb"))
    pickle.dump(y_train, open("data/y_train", "wb"))
    pickle.dump(tokenizer.word_index,
                open(f"data/word_index_spell{with_spell}", "wb"))

    return x_train_encoded_padded_sequences, x_test_encoded_padded_sequences, y_train, tokenizer.word_index


def read_embeddings():
    logger.info("Reading embeddings")

    EMBEDDING_FILE_FASTTEXT = "data/crawl-300d-2M.vec"
    EMBEDDING_FILE_TWITTER = "data/glove.twitter.27B.200d.txt"

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index_ft = dict(
        get_coefs(*o.rstrip().rsplit(' '))
        for o in open(EMBEDDING_FILE_FASTTEXT, encoding='utf-8'))
    embeddings_index_tw = dict(
        get_coefs(*o.strip().split())
        for o in open(EMBEDDING_FILE_TWITTER, encoding='utf-8'))
    return embeddings_index_ft, embeddings_index_tw


def index_to_embedding(fast_text, glove, word_index):
    logger.info("Creating embedding matrix using word to index mapping")
    nb_words = len(word_index) + 1
    embedding_matrix = np.zeros((nb_words, 501))

    OUT_OF_VOCABULARY = np.zeros((501, ))

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
            f += 1
        else:
            embedding_matrix[i] = OUT_OF_VOCABULARY
            not_found += 1
    logging.info(
        "Words in sequences: %i, emdedded: %i, not found: %i into embedding maxtrix of shaape %s",
        len(word_index), f, not_found, embedding_matrix.shape)
    return embedding_matrix


def create_embedding_matrix(word_index, with_spell=False, from_picked=True):
    if from_picked:
        return pickle.load(open(f"data/matrix_spell_{with_spell}", "rb"))

    fast_text, glove = read_embeddings()
    matrix = index_to_embedding(fast_text, glove, word_index)
    pickle.dump(matrix, open(f"data/matrix_spell_{with_spell}", "wb"))

    return matrix


def preprocess(from_pickle=True, with_spell=False, prototyping=True):

    x_train, x_test, y_train, word_index = encode_sequences(
        protopying=prototyping, from_pickle=from_pickle, with_spell=with_spell)
    matrix = create_embedding_matrix(word_index, with_spell, from_pickle)
    return x_train, x_test, y_train, matrix

def bert_sequence_embeddings():
    '''train and test sequences embedded with bert as a service, first 200 tokens
    one sequence results in one embedding
    - encoding train and test took 10h each on a 4 core laptop (I would not do that again:)
    '''
    embeddings = np.load(open("data/embeddings_200", "rb"))
    test_embeddings = np.load(open("data/test_embeddings_200", "rb"))
    return embeddings, test_embeddings

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    preprocess(from_pickle=False, prototyping=True)
