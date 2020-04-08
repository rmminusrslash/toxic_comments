import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from model import get_model
from preprocess import preprocess


def train():
    embeddings = np.load(open("data/embeddings_200", "rb"))
    test_embeddings = np.load(open("data/test_embeddings_200", "rb"))
    targets = np.load(open("data/targets_200", "rb"))

    callbacks = [
        ReduceLROnPlateau(),
        EarlyStopping(patience=4),
        ModelCheckpoint(filepath='model-trained_embedding', monitor="val_auc", mode="max", save_best_only=True,
                        verbose=1)
    ]

    # X_train, X_test, y_train, y_test = train_test_split(embeddings, targets, test_size=0.1, random_state=15)
    prototyping = False
    if prototyping:
        model = get_model(np.zeros((1000, 501)))
        x_train = np.array([[23] * 900] * 10)  # batchsize x seq_lenght
        embeddings = np.array([[0.2] * 768] * 10)  # batchsize x bert_embeddings
        y_train = np.array([[1, 0, 1, 0, 0, 1]] * 10)
    else:

        x_train, x_test, y_train, matrix = preprocess()
        model = get_model(matrix, with_fasttext=True, train_embeddings=True)

    history = model.fit([x_train, embeddings], y_train,
                        class_weight=None,
                        epochs=20,
                        batch_size=512,
                        validation_split=.1,
                        callbacks=callbacks,
                        verbose=1)

    model.summary()




if __name__ == "__main__":
    #matrix = create_embedding_matrix(None, from_picked=True)
    #print(matrix.shape)
    model = get_model(np.zeros((1,1)), with_fasttext=True)
    submit(model, "model-embeddings-not-trainable-with-spell", prototyping=False)
    import os

    description = "Bert  fixed, fasttext +glove fixed, with spell, feed forward network"
    os.system(
        'kaggle competitions submit -c jigsaw-toxic-comment-classification-challenge -f submission.csv -m "%s"' % description)
