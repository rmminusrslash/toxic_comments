# This code is  based on Peter Norvig's spell checker and adapted using ranks instead of frequencies
# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec
import logging

import gensim

logger = logging.getLogger("spellcorrection")


class SpellCorrect():
    def __init__(self, case_sensitive=True, words_by_popularity_desc=None):
        '''
        words_by_popularity_desc is a list of words sorted by popularity descending
        '''
        self.maybe_to_lower = (lambda word: word) if case_sensitive else (
            lambda word: word.lower())
        self.wordToRank = self.create_word_to_rank(words_by_popularity_desc)
        self.corrected = 0
        self.uncorrected = 0

    def create_word_to_rank(self, words_by_popularity_desc=None):
        if words_by_popularity_desc is None:
            logger.info("Reading dictionary for spell correction")
            EMBEDDING_FILE_FASTTEXT = "data/crawl-300d-2M.vec"
            words_by_popularity_desc = gensim.models.KeyedVectors.load_word2vec_format(
                EMBEDDING_FILE_FASTTEXT).index2word
            logger.info("Loaded %s words", len(words_by_popularity_desc))
        w_rank = {}
        for i, word in enumerate(words_by_popularity_desc):
            w_rank[self.maybe_to_lower(word)] = i
        return w_rank

    def P(self, word):
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return -self.wordToRank.get(word, 0)

    def correction(self, word):
        "Most probable spelling correction for word."
        if word in self.wordToRank:
            return word
        corrected = max(self.candidates(self.maybe_to_lower(word)), key=self.P)
        self.corrected += word != corrected
        self.uncorrected += 1
        if corrected != word:
            print(word, corrected)
        else:
            corrected = max(self.candidates(
                self.singlify(self.maybe_to_lower(word))),
                            key=self.P)
            if corrected != word:
                print("SINGLYFY", word, corrected)
        return corrected

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word))
                or self.known(self.edits2(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.wordToRank)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def singlify(self, word):
        return "".join([
            letter for i, letter in enumerate(word)
            if i == 0 or letter != word[i - 1]
        ])


SpellCorrect().correction("need")
