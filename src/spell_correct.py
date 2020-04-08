# This code is  based on: Spellchecker using Word2vec by CPMP
# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec
import pickle
import re
import gensim


class SpellCorrect():

    def __init__(self, word_rank_from_pickle=True):
        self.wordToRank = self.read_word_ranker(word_rank_from_pickle)

    def read_word_ranker(self, word_rank_from_pickle):
        if word_rank_from_pickle:
            spell_model = pickle.load(open("data/spell_model", "rb"))
        else:
            EMBEDDING_FILE_FASTTEXT = "data/crawl-300d-2M.vec"
            spell_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE_FASTTEXT, limit=300000)
        words = spell_model.index2word
        w_rank = {}
        for i, word in enumerate(words):
            w_rank[word] = i
        return w_rank

    # Use fast text as vocabulary
    def words(self, text):
        return re.findall(r'\w+', text.lower())

    def P(self, word):
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.wordToRank.get(word, 0)

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

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
        return "".join([letter for i, letter in enumerate(word) if i == 0 or letter != word[i - 1]])
