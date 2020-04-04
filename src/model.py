import keras
import numpy as np
import pandas as pd
from keras import optimizers, Model
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, Input, concatenate
from sklearn.metrics import roc_auc_score

from look_ahead import LookaheadOptimizerCallback
from preprocess import preprocess


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_better_count = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            if (score > self.max_score):
                print("*** New High Score (previous: %.6f) \n" % self.max_score)
                model.save_weights("best_weights.h5")
                self.max_score=score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > 3:
                    print("Epoch %05d: early stopping, high score = %.6f" % (epoch,self.max_score))
                    self.model.stop_training = True


def get_model(embedding_matrix, with_fasttext=False, maxlen=900,train_embeddings=False):
    sequences_indexed = Input(shape=(maxlen,))
    bert_embeddings = Input(shape=(768,))

    # embed sequences keras
    sequences_embedded = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                                   weights=[embedding_matrix], trainable=train_embeddings)(sequences_indexed)
    # the model will take as input an integer matrix of size (batch, input_length).
    # the largest integer (i.e. word index) in the input should be input_dim -1 (vocabsize)
    # now model.output_shape == (None, 10, 64), where None is the batch dimension.

    # TODO: masking?
    # toDO: add max?
    # flatten?
    # todo: trainable=True?

    if with_fasttext:
        sequences_average = keras.layers.GlobalAveragePooling1D(data_format="channels_last")(sequences_embedded)
        sequences_max = keras.layers.GlobalMaxPool1D(data_format="channels_last")(sequences_embedded)
        # channels_last` corresponds to inputs with shape batch, steps, features)

        x = concatenate([sequences_average, sequences_max, bert_embeddings])

        layer_1 = Dense(512, activation='relu')(x)
    else:
        layer_1 = Dense(512, activation='relu')(bert_embeddings)

    num_classes_classes = 6
    predictions = Dense(num_classes_classes, activation='sigmoid')(layer_1)

    model = Model(inputs=[sequences_indexed, bert_embeddings], outputs=predictions)
    adam = optimizers.adam()

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc')])


    return model

def submit(model, checkpoint_dir, prototyping=False):
    model.load_weights(checkpoint_dir)
    print(model.summary())

    test_labels_path = 'data/test_labels.csv'
    df_test_labels = pd.read_csv(test_labels_path)

    if prototyping:
        x_test = np.array([[23] * 900] * 10)  # batchsize x seq_lenght
        test_embeddings = np.array([[0.2] * 768] * 10)  # batchsize x bert_embeddings
        df_test_labels = df_test_labels[:10]
    else:
        x_train, x_test, y_train, matrix = preprocess()
        test_embeddings = np.load(open("data/test_embeddings_200", "rb"))

    test_predictions = model.predict([x_test, test_embeddings], verbose=2)

    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predictions = pd.DataFrame(test_predictions, columns=label_cols).set_index(df_test_labels.id)

    predictions.to_csv("submission.csv")

if __name__ == "__main__":

    embeddings = np.load(open("data/embeddings_200", "rb"))
    test_embeddings = np.load(open("data/test_embeddings_200", "rb"))
    targets = np.load(open("data/targets_200", "rb"))

    # X_train, X_test, y_train, y_test = train_test_split(embeddings, targets, test_size=0.1, random_state=15)

    for tr in [False, False]:
        lookahead_callback = LookaheadOptimizerCallback()
        callbacks = [
            lookahead_callback,
            EarlyStopping(patience=4, monitor="val_auc", mode="max"),
            ModelCheckpoint(filepath='model', monitor="val_auc", mode="max", save_best_only=True, verbose=1)
        ]

        prototyping = False
        if prototyping:
            model = get_model(np.zeros((1000, 501)))
            x_train = np.array([[23] * 900] * 10)  # batchsize x seq_lenght
            embeddings = np.array([[0.2] * 768] * 10)  # batchsize x bert_embeddings
            y_train = np.array([[1, 0, 1, 0, 0, 1]] * 10)
        else:

            x_train, x_test, y_train, matrix = preprocess(True)
            model = get_model(matrix, with_fasttext=True, train_embeddings=tr)



        history = model.fit([x_train, embeddings], y_train,
                            class_weight=None,
                            epochs=20,
                            batch_size=512,
                            validation_split=.1,
                            callbacks=callbacks,
                            verbose=1)

        model.summary()

        submit(model, "model", prototyping=False)
        import os

        description = "Bert  fixed, fasttext +glove trainable=%s, with spell, feed forward network" % tr
        os.system(
            'kaggle competitions submit -c jigsaw-toxic-comment-classification-challenge -f submission.csv -m "%s"' % description)



