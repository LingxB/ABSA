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

datapath = 'Data\OpeNER\OpeNER_hotel_en.csv'
dataset = pd.read_csv(datapath)


# Preprocessing
def input_ready(df, mlen):
    df['TOKENS'] = df.OEXP.apply(lambda x: x.split())
    df['TLEN'] = df.TOKENS.apply(lambda x: len(x))
    df['CLASS'] = ~df.POLARITY.str.contains('Negative')
    c = freq_dist(df.TOKENS)
    w_idx = w_index(c, start_idx=1)
    config.update({'vocab':len(w_idx)})
    data = df2feats(df, 'TOKENS', w_idx)
    X = sequence.pad_sequences(data, maxlen=mlen).astype('float32')
    y = df.CLASS.values.astype('float32')
    return X,y

data,labels = input_ready(dataset, config['time_steps'])
X_train,X_test = train_test(data)
y_train,y_test = train_test(labels)

adam = Adam(lr=config['learning_rate'], decay=config['decay'])
model = Sequential()
model.add(Embedding(config['vocab']+2, config['embedding_size'], input_length=config['time_steps'], mask_zero=True))
model.add(Bidirectional(LSTM(config['LSTM_cell'], dropout=config['drop_out'], recurrent_dropout=config['drop_out'])))
#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(config['drop_out']))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
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


# Train on 3112 samples, validate on 1038 samples
# Epoch 1/15
# 4s - loss: 0.5188 - acc: 0.7812 - val_loss: 0.4096 - val_acc: 0.8025
# Epoch 2/15
# 1s - loss: 0.2085 - acc: 0.9319 - val_loss: 0.3746 - val_acc: 0.8198
# Epoch 3/15
# 1s - loss: 0.1130 - acc: 0.9669 - val_loss: 0.4797 - val_acc: 0.8353
# Epoch 4/15
# 1s - loss: 0.0868 - acc: 0.9733 - val_loss: 0.5090 - val_acc: 0.8304
# Epoch 5/15
# 1s - loss: 0.0710 - acc: 0.9791 - val_loss: 0.5924 - val_acc: 0.8353
# Epoch 6/15
# 1s - loss: 0.0608 - acc: 0.9794 - val_loss: 0.6348 - val_acc: 0.8324
# Epoch 7/15
# 1s - loss: 0.0536 - acc: 0.9817 - val_loss: 0.6558 - val_acc: 0.8179
# Epoch 8/15
# 1s - loss: 0.0493 - acc: 0.9843 - val_loss: 0.7046 - val_acc: 0.8218
# Epoch 9/15
# 1s - loss: 0.0471 - acc: 0.9826 - val_loss: 0.6436 - val_acc: 0.8198
# Epoch 10/15
# 1s - loss: 0.0456 - acc: 0.9830 - val_loss: 0.6862 - val_acc: 0.8266
# Epoch 11/15
# 2s - loss: 0.0405 - acc: 0.9843 - val_loss: 0.7600 - val_acc: 0.8247
# Epoch 12/15
# 1s - loss: 0.0386 - acc: 0.9849 - val_loss: 0.7789 - val_acc: 0.8276
# Epoch 13/15
# 1s - loss: 0.0366 - acc: 0.9862 - val_loss: 0.7873 - val_acc: 0.8218
# Epoch 14/15
# 2s - loss: 0.0330 - acc: 0.9875 - val_loss: 0.7989 - val_acc: 0.8256
# Epoch 15/15
# 2s - loss: 0.0328 - acc: 0.9859 - val_loss: 0.7912 - val_acc: 0.8266
# Config: {'batch_size': 32, 'learning_rate': 0.001, 'decay': 0.0, 'embedding_size': 200, 'LSTM_cell': 64, 'time_steps': 10, 'vocab': 1964, 'drop_out': 0.2, 'epochs': 15}
# Test loss: 0.791201767205
# Test accuracy: 0.826589595491
