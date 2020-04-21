'''
Baseline implementation for text classification applied to the kaggle challenge "Toxic Comments"
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/.

Algorithm:

Each document is represented as a set of words ("bag of words"). Each word in the document is represented as a numeric
value. There are two options how the value is calculated:
i) the value is the naive bayes probability for a class given the word, e.g. p(class=toxix|word=fucker)=15%.
ii) same as i), but instead of a binary occurrence per word we use the tf-idf values of the word in the document
These numeric features are then input into a logistic regression. A separate model is learned per class.

The baseline follows the paper "Baselines and Bigrams: Simple, Good Sentiment and Topic Classiﬁcation" and
the improvements from Jeremy Howard done in https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline.
The baseline's hyperparameters were tuned further to achieve additional improvements.

Experiment results
The baseline achieves a class averaged AUC of 0.978. State of the art models like Bert achieve 0.985
as evidenced by the publicly available notebooks in the competition. The challenge winners achieved 0.989 and used
recurrent models and stacking. The best single single model achieved 0.987 and is also a recurrent neural network.
Please refer to the readme for a more detailed performance discussion.

Runtime:
The tf-idf version takes 1 min(!) to run on my laptop for the model with the best hyperparameters. A training and
prediction run of bert takes 60 minutes on Kaggle's GPU NVIDIA TESLA P100.

Overall recommendation:
This baseline is very strong in performance, low in training time and easy to interpret. It is surprising how close 
such a simple baseline comes to state of the art models. It can be used as a fast bootstrapping technique.

'''

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

logger = logging.getLogger(__file__)


def train_baseline():
    logger.info("start training")

    binary = False
    n_grams = 1  # 1-gram performed better or equal than bi-grams, original notebook=2
    regularization = 0.2  # more regularization performed, original notebook: C=4
    log_level_training = 0  # replace with 1 for more verbosity
    binary_conditional_prob = False
    submit = False

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv("data/test.csv")

    import re, string
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    def tokenize(s):
        # replaces matched letters with space
        return re_tok.sub(r' \1 ', s).split()

    if binary:
        # document x word matrix
        counter = CountVectorizer(ngram_range=(1, n_grams),
                                  binary=True,
                                  min_df=3,
                                  tokenizer=tokenize,
                                  max_df=0.9,
                                  strip_accents='unicode')
    else:
        counter = TfidfVectorizer(
            ngram_range=(1, n_grams),
            tokenizer=tokenize,
            min_df=
            3,  # word must appear in at least min_df documents to be considered
            max_df=
            0.9,  # words that appear in 90% of the documents are not considered
            strip_accents='unicode',
            use_idf=1,
            smooth_idf=1,
            sublinear_tf=1)

    word_occurence = counter.fit_transform(
        train["comment_text"])  # comments x words
    word_occurence_test = counter.transform(
        test["comment_text"])  # comments x words
    logger.info("Found %i words, sparse matrix of size %i",
                len(counter.get_feature_names()), word_occurence.data.size)

    # k fold run, advantages:
    # 1. more true representation of performance by looking at validation metrics of multiple runs
    # 2. allows to use 100% of training data to predict the testset (not relevant here, mostly important for methods
    #  that do model selection on the validation set like deep learning)
    # 3. slight performance uplift by averaging predictions of k runs instead of a single run

    num_folds = 3  # number of folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)

    label_cols = [
        'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    ]
    y_train = train[label_cols].values
    test_predict = np.zeros((len(test), len(label_cols)))
    metrics = dict([(c, []) for c in label_cols])

    for train_index, val_index in kf.split(word_occurence):
        logger.info("Starting training fold")
        kfold_y_train, kfold_y_val = y_train[train_index], y_train[val_index]
        kfold_x_train, kfold_x_val = word_occurence[
            train_index], word_occurence[val_index]

        # learning model for each class
        for c in range(len(label_cols)):  # change to 6 when done prototyping

            logger.info("Learning model for %s", label_cols[c])

            def pr(word_occurence, y, class_index):
                # conditional probabilites for a class given a word
                class_c = word_occurence[y == class_index]  # doc_c x words
                if binary_conditional_prob:
                    class_c = class_c > 0
                word_probability = (np.sum(class_c, axis=0) +
                                    1) / (class_c.shape[0] + 1)
                return word_probability  # 1x words

            y = kfold_y_train[:, c]
            r = np.log(pr(kfold_x_train, y, 1) / pr(kfold_x_train, y, 0))

            naive_bayes_features = kfold_x_train.multiply(r)  # docs x words

            log = LogisticRegression(C=regularization,
                                     dual=True,
                                     solver="liblinear",
                                     max_iter=5000,
                                     verbose=log_level_training)

            log.fit(naive_bayes_features, y)

            y_val = kfold_y_val[:, c]
            naive_bayes_features_val = kfold_x_val.multiply(r)
            # sklearn's predict proba returns the probabilities for label 0 and label 1 in two separate dimensions
            y_val_predict = log.predict_proba(naive_bayes_features_val)[:, 1]
            auc_val = roc_auc_score(y_val, y_val_predict)
            auc_train = roc_auc_score(
                y,
                log.predict_proba(naive_bayes_features)[:, 1])

            metrics[label_cols[c]].append(auc_val)

            logger.info('auc train %.4f, auc val: %.4f', auc_train, auc_val)

            naive_bayes_features_test = word_occurence_test.multiply(
                r)  # docs x words
            # test predictions are averaged over the k runs
            test_predict[:, c] += log.predict_proba(
                naive_bayes_features_test)[:, 1] / num_folds

    for c in metrics.keys():
        logger.info("%s average validation auc %.4f ", c, np.mean(metrics[c]))
    logger.info("overall %.4f", np.mean(list(metrics.values())))

    if submit:
        import kaggle

        logger.info("Writing submission file")
        result = pd.read_csv("data/test_labels.csv")
        result[label_cols] = test_predict
        result.to_csv('baseline_submission.csv', index=False)

        description = "Baseline, logReg with %s naive bayes features, %i-gram, reguarization C=%s" % (
            "binary" if binary else "tf-idf", n_grams, regularization)

        kaggle.api.competition_submit(
            "baseline_submission.csv", description,
            "jigsaw-toxic-comment-classification-challenge")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    train_baseline()
