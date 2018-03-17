import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import re
from math import floor
import sys

def tokenize_corpus(sents_lines):
    for l in range(len(sents_lines)):
        sents_lines[l] = re.sub(r'[^a-zA-Z ]+', '', sents_lines[l])
    sents_lines = [re.sub(r'[^a-zA-Z ]+', '', s).split(' ') for s in sents_lines]
    sents_vector = [list(filter(lambda x: len(x), s)) for s in sents_lines]

    return sents_vector

def token_to_vector(corpus_tokens, embedding_model, max_post_length):
    word2idx = {"_PAD": 0} #initialize dict
    vocab_list = [(k, embedding_model.wv[k]) for k, v in embedding_model.wv.vocab.items()] 
    
    embeddings_matrix = np.zeros((len(embedding_model.wv.vocab.items()) + 1, embedding_model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]
        
    corpus_idx = [[word2idx[w] if w in word2idx else 0 for w in s]for s in corpus_tokens]
    corpus_with_idx = sequence.pad_sequences(corpus_idx, maxlen=max_post_length, padding='post')
    
    return corpus_with_idx


#load testing data
testing_lines = open(sys.argv[1]).readlines()
testing_lines = [l[l.index(',')+1:].replace('\n','') for l in testing_lines][1:]
testing_sents_tokens = tokenize_corpus(testing_lines)



#load word2vec model
w2v_model1 = Word2Vec.load('model_w2v_all.bin')
w2v_model2 = Word2Vec.load('model_w2v_all_test_200')

#convert to idx
corpus_with_idx_test1 = token_to_vector(testing_sents_tokens, w2v_model1, 55)
corpus_with_idx_test2 = token_to_vector(testing_sents_tokens, w2v_model2, 30)

#load model
model1 = load_model('model/rnn02_semi_819') #w2v_model1
model2 = load_model('model/biLSTM3_8191')


#make prediction
pred1 = model1.predict(corpus_with_idx_test1, verbose=1)
pred2 = model2.predict(corpus_with_idx_test2, verbose=1)

pred1 = np.array(pred1.flatten())
pred2 = np.array(pred2.flatten())



#ensemble all prediction
preds_sum = np.sum([pred1,pred2], axis=0)
predict_probs = preds_sum/2
predict_answer = [0 if i < 0.5 else 1 for i in predict_probs]


#submission form
submission_data = pd.read_csv("data/sampleSubmission.csv",sep=',',header=0)
out_df = submission_data.copy()
out_df['label'] = predict_answer
out_df.to_csv(sys.argv[2],index=None)





