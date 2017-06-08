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

datapath = 'Data\OpeNER\OpeNER_hotel_es.csv'
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


# Train on 3291 samples, validate on 1097 samples
# Epoch 1/15
# 4s - loss: 0.4995 - acc: 0.8232 - val_loss: 0.3568 - val_acc: 0.8624
# Epoch 2/15
# 1s - loss: 0.2008 - acc: 0.9398 - val_loss: 0.3263 - val_acc: 0.8596
# Epoch 3/15
# 2s - loss: 0.0901 - acc: 0.9720 - val_loss: 0.3548 - val_acc: 0.8624
# Epoch 4/15
# 2s - loss: 0.0639 - acc: 0.9833 - val_loss: 0.3982 - val_acc: 0.8678
# Epoch 5/15
# 2s - loss: 0.0523 - acc: 0.9869 - val_loss: 0.4088 - val_acc: 0.8532
# Epoch 6/15
# 2s - loss: 0.0476 - acc: 0.9845 - val_loss: 0.4431 - val_acc: 0.8532
# Epoch 7/15
# 2s - loss: 0.0421 - acc: 0.9878 - val_loss: 0.4418 - val_acc: 0.8614
# Epoch 8/15
# 1s - loss: 0.0382 - acc: 0.9878 - val_loss: 0.4577 - val_acc: 0.8642
# Epoch 9/15
# 2s - loss: 0.0365 - acc: 0.9894 - val_loss: 0.4823 - val_acc: 0.8487
# Epoch 10/15
# 1s - loss: 0.0335 - acc: 0.9903 - val_loss: 0.4931 - val_acc: 0.8669
# Epoch 11/15
# 2s - loss: 0.0308 - acc: 0.9897 - val_loss: 0.4822 - val_acc: 0.8624
# Epoch 12/15
# 2s - loss: 0.0274 - acc: 0.9912 - val_loss: 0.5164 - val_acc: 0.8469
# Epoch 13/15
# 2s - loss: 0.0259 - acc: 0.9906 - val_loss: 0.5261 - val_acc: 0.8614
# Epoch 14/15
# 2s - loss: 0.0233 - acc: 0.9912 - val_loss: 0.5461 - val_acc: 0.8596
# Epoch 15/15
# 2s - loss: 0.0207 - acc: 0.9909 - val_loss: 0.5419 - val_acc: 0.8624
# Config: {'decay': 0.0, 'vocab': 2082, 'LSTM_cell': 64, 'embedding_size': 200, 'epochs': 15, 'drop_out': 0.2, 'time_steps': 10, 'learning_rate': 0.001, 'batch_size': 32}
# Test loss: 0.541891769865
# Test accuracy: 0.862351868787

