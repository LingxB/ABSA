from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.optimizers import Adam
from keras import backend as K
import pandas as pd
import numpy as np
from src.utils.preprocess import *


config = {'embedding_size': 200,
          'time_steps': 10,
          'drop_out': 0.2,
          'LSTM_cell': 64,
          'learning_rate': 0.001,
          'decay': 0.0,
          'batch_size': 32,
          'epochs': 15
         }

train_path = 'Data\OpeNER\PL\OpeNER_hotel_es_train.csv'
test_path = 'Data\OpeNER\PL\OpeNER_hotel_es_test.csv'
datapath = 'Data\OpeNER\OpeNER_hotel_es.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
dataset = pd.read_csv(datapath)



# Preprocessing
def global_stats(df):
    df['TOKENS'] = df.OEXP.apply(lambda x: x.split())
    df['TLEN'] = df.TOKENS.apply(lambda x: len(x))
    df['CLASS'] = ~df.POLARITY.str.contains('Negative')
    c = freq_dist(df.TOKENS)
    w_idx = w_index(c, start_idx=1)
    config.update({'vocab':len(w_idx)})
    return w_idx

def input_ready(df, mlen, w_idx):
    df['TOKENS'] = df.OEXP.apply(lambda x: x.split())
    df['TLEN'] = df.TOKENS.apply(lambda x: len(x))
    df['CLASS'] = ~df.POLARITY.str.contains('Negative')

    data = df2feats(df, 'TOKENS', w_idx)
    X = sequence.pad_sequences(data, maxlen=mlen).astype('float32')
    y = pd.get_dummies(df.POLARITY).values.astype('float32')
    return X,y

w_idx = global_stats(dataset)
X_train,y_train = input_ready(train, config['time_steps'], w_idx)
X_test,y_test = input_ready(test, config['time_steps'], w_idx)


# data,labels = input_ready(dataset, config['time_steps'])
# X_train,X_test = train_test(data)
# y_train,y_test = train_test(labels)

adam = Adam(lr=config['learning_rate'], decay=config['decay'])
model = Sequential()
model.add(Embedding(config['vocab']+2, config['embedding_size'], input_length=config['time_steps'], mask_zero=True))
model.add(Bidirectional(LSTM(config['LSTM_cell'], dropout=config['drop_out'], recurrent_dropout=config['drop_out'])))
#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(config['drop_out']))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train,
          batch_size=config['batch_size'],
          epochs=config['epochs'],
          verbose=2,
          validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test,
                           batch_size=config['batch_size'],
                           verbose=2)

print('\nConfig:', config)
print('Test loss:', loss)
print('Test accuracy:', acc)


# Train on 3906 samples, validate on 482 samples
# Epoch 1/15
# 19s - loss: 1.0132 - acc: 0.6905 - val_loss: 0.7529 - val_acc: 0.7137
# Epoch 2/15
# 15s - loss: 0.4373 - acc: 0.8571 - val_loss: 0.5902 - val_acc: 0.7697
# Epoch 3/15
# 15s - loss: 0.2298 - acc: 0.9355 - val_loss: 0.6412 - val_acc: 0.7697
# Epoch 4/15
# 15s - loss: 0.1622 - acc: 0.9524 - val_loss: 0.5964 - val_acc: 0.7946
# Epoch 5/15
# 17s - loss: 0.1337 - acc: 0.9552 - val_loss: 0.6644 - val_acc: 0.7822
# Epoch 6/15
# 16s - loss: 0.1172 - acc: 0.9606 - val_loss: 0.6466 - val_acc: 0.7884
# Epoch 7/15
# 15s - loss: 0.1066 - acc: 0.9642 - val_loss: 0.7062 - val_acc: 0.7780
# Epoch 8/15
# 15s - loss: 0.0984 - acc: 0.9665 - val_loss: 0.7118 - val_acc: 0.7697
# Epoch 9/15
# 14s - loss: 0.0924 - acc: 0.9677 - val_loss: 0.7414 - val_acc: 0.7780
# Epoch 10/15
# 18s - loss: 0.0945 - acc: 0.9695 - val_loss: 0.7908 - val_acc: 0.7697
# Epoch 11/15
# 16s - loss: 0.0982 - acc: 0.9659 - val_loss: 0.7603 - val_acc: 0.7759
# Epoch 12/15
# 15s - loss: 0.0794 - acc: 0.9713 - val_loss: 0.7713 - val_acc: 0.7780
# Epoch 13/15
# 16s - loss: 0.0772 - acc: 0.9734 - val_loss: 0.8134 - val_acc: 0.7676
# Epoch 14/15
# 15s - loss: 0.0733 - acc: 0.9747 - val_loss: 0.8029 - val_acc: 0.7884
# Epoch 15/15
# 15s - loss: 0.0728 - acc: 0.9759 - val_loss: 0.8268 - val_acc: 0.7801
# Config: {'embedding_size': 200, 'decay': 0.0, 'drop_out': 0.2, 'learning_rate': 0.001, 'vocab': 2082, 'batch_size': 32, 'time_steps': 10, 'epochs': 15, 'LSTM_cell': 64}
# Test loss: 0.826770704589
# Test accuracy: 0.780082987552



