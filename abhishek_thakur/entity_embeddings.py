import os
import gc
import joblib
import pandas as pd
import numpy as np

from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils


def create_model(data, catacols):
    '''

    :param data:
    :param catacols:
    :return:
    '''
    # initializing list of inputs
    inputs = []

    # initializing list of outputs
    outputs = []

    for col in catacols:
        # find the number of unique values in the column
        num_unique_vals = int(data[col].nunique())

        # simple dimension of embedding calculator
        # min. size is half of the number of unique values
        # max. size is 50. max size depends on the number of unique
        # categories too. Usually, 50 is quite sufficient but if you have
        # millions of unique values, you might need a larger dimension.
        embed_dim = int(min(np.ceil(num_unique_vals/2), 50))

        # simple keras input layer with size 1
        inp = layers.Input(shape=(1,))

        # add embedding layer to raw input
        # embedding size is always 1 more than unique values in input
        out = layers.Embedding(
            num_unique_vals + 1, embed_dim, name=col
        )(inp)

        # 1-d spatial dropout is the standard for embedding layers
        #  you can use it in NLP tasks too.
        out = layers.SpatialDropout1D(0.3)(out)

        # reshape the input to the dimension of embedding
        # this becomes our output layer for current feature
        out = layers.Reshape(target_shape=(embed_dim, ))(out)

        # add input to input list
        inputs.append(inp)

        # add output to output list
        outputs.append(out)

    # concatenate all output layers
    x = layers.Concatenate()(outputs)

    # add a batchnorm layer.
    x = layers.BatchNormalization()(x)

    # a bunch of dense layers with dropout.
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    # using softmax and treating it as a two class problem
    # you can also use sigmoid, then u need to use only one output class
    y = layers.Dense(2, activation='softmax')(x)

    # create final model
    model = Model(inputs=inputs, outputs=y)

    # compile the model
    # we use adam and cross entropy
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def run(_fold):
    '''

    :param _fold:
    :return:
    '''

    # load the cat_in_the_dat data
    df = pd.read_csv('./data/cat_in_the_dat_train_folds.csv')

    # list of all input features/predictors.
    features = [col
                for col in df.columns
                if col not in ['id', 'target', 'kfold']
                ]

    # initializing label encoder
    lbl = preprocessing.LabelEncoder()

    # fill all the numm values with NONE
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')
        df.loc[:, col] = lbl.fit_transform(df[col])

    # train and test data
    df_train = df.loc[df['kfold'] != _fold, :]
    df_valid = df.loc[df['kfold'] == _fold, :]

    model = create_model(df, features)

    # our features are list of lists
    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]
    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]

    #fetch target columns
    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    # convert target columns to categories
    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    # fit the model
    model.fit(xtrain,
              ytrain_cat,
              validation_data=(xvalid, yvalid_cat),
              verbose=1,
              batch_size=1024,
              epochs=3
              )

    # generate validation predictions
    valid_preds = model.predict(xvalid)[:, 1]

    # print roc auc score
    auc = metrics.roc_auc_score(yvalid, valid_preds)

    print(f'fold: {_fold}, auc: {auc}')

    # clear session to free up GPU memory
    K.clear_session()

    return None


if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)


