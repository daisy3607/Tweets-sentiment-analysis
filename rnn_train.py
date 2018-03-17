import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Bidirectional, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import re
from math import floor
import keras
import sys

def tokenize_corpus(sents):
    for l in range(len(sents)):
        sents[l] = re.sub(r'[^a-zA-Z ]+', '', sents[l])
    sents = [re.sub(r'[^a-zA-Z ]+', '', s).split(' ') for s in sents]
    sents_vector = [list(filter(lambda x: len(x), s)) for s in sents]
    return sents_vector

def split_valid_data(train_x, train_y, percentage):
    all_data_size = len(train_x)
    valid_data_size = int(floor(all_data_size * percentage))
    x_train, y_train = train_x[valid_data_size:], train_y[valid_data_size:]
    x_valid, y_valid = train_x[0:valid_data_size], train_y[0:valid_data_size]

    return x_train, y_train, x_valid, y_valid

#load data
lines = open(sys.argv[1]).readlines()
label = [int(l[0]) for l in lines] 
sents = [l[10:].replace('\n','') for l in lines]
sents_tokens = tokenize_corpus(sents)


#load word2vec model
w2v_model = Word2Vec.load('model_w2v_all_test_200')

#build a dict
word2idx = {"_PAD": 0} #initialize dict
vocab_list = [(k, w2v_model.wv[k]) for k, v in w2v_model.wv.vocab.items()] 

#word embedding
embeddings_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embeddings_matrix[i + 1] = vocab_list[i][1]

#training parameters
max_post_length = 30
EMBEDDING_DIM = 200 
batch_size = 128
epochs = 4

#embedding layer
embedding_layer = Embedding(len(embeddings_matrix),
                            EMBEDDING_DIM,
                            weights=[embeddings_matrix],
                            trainable=False)

#corpus id
corpus_idx = [[word2idx[w] if w in word2idx else 0 for w in s]for s in sents_tokens]

#corpus
corpus_with_idx = sequence.pad_sequences(corpus_idx, maxlen=max_post_length, padding='post')

# #split valid data
split_percentage = 0.1
X_train, Y_train, X_valid, Y_valid = split_valid_data(corpus_with_idx, label, split_percentage)


#build model
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(512)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
saveBestModel=keras.callbacks.ModelCheckpoint('model/model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[earlyStopping,saveBestModel], 
          verbose=1,
          validation_data=(X_valid, Y_valid))

