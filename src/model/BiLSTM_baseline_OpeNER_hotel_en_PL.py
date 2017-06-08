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

train_path = 'Data\OpeNER\PL\OpeNER_hotel_en_train.csv'
test_path = 'Data\OpeNER\PL\OpeNER_hotel_en_test.csv'
datapath = 'Data\OpeNER\OpeNER_hotel_en.csv'
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
    y = df.CLASS.values.astype('float32')
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


# Train on 3643 samples, validate on 507 samples
# Epoch 1/15
# 4s - loss: 0.5017 - acc: 0.7867 - val_loss: 0.3154 - val_acc: 0.8639
# Epoch 2/15
# 2s - loss: 0.2051 - acc: 0.9358 - val_loss: 0.3124 - val_acc: 0.8876
# Epoch 3/15
# 2s - loss: 0.1174 - acc: 0.9621 - val_loss: 0.3423 - val_acc: 0.8817
# Epoch 4/15
# 2s - loss: 0.0956 - acc: 0.9668 - val_loss: 0.3354 - val_acc: 0.8876
# Epoch 5/15
# 2s - loss: 0.0808 - acc: 0.9717 - val_loss: 0.3753 - val_acc: 0.8817
# Epoch 6/15
# 2s - loss: 0.0692 - acc: 0.9772 - val_loss: 0.4361 - val_acc: 0.8836
# Epoch 7/15
# 2s - loss: 0.0634 - acc: 0.9789 - val_loss: 0.4606 - val_acc: 0.8817
# Epoch 8/15
# 2s - loss: 0.0573 - acc: 0.9789 - val_loss: 0.4783 - val_acc: 0.8738
# Epoch 9/15
# 2s - loss: 0.0520 - acc: 0.9811 - val_loss: 0.5175 - val_acc: 0.8797
# Epoch 10/15
# 2s - loss: 0.0508 - acc: 0.9808 - val_loss: 0.4918 - val_acc: 0.8679
# Epoch 11/15
# 2s - loss: 0.0472 - acc: 0.9824 - val_loss: 0.5177 - val_acc: 0.8797
# Epoch 12/15
# 2s - loss: 0.0426 - acc: 0.9849 - val_loss: 0.5658 - val_acc: 0.8698
# Epoch 13/15
# 2s - loss: 0.0403 - acc: 0.9841 - val_loss: 0.5719 - val_acc: 0.8619
# Epoch 14/15
# 2s - loss: 0.0401 - acc: 0.9844 - val_loss: 0.5755 - val_acc: 0.8679
# Epoch 15/15
# 2s - loss: 0.0378 - acc: 0.9863 - val_loss: 0.5680 - val_acc: 0.8738
# Config: {'epochs': 15, 'drop_out': 0.2, 'learning_rate': 0.001, 'LSTM_cell': 64, 'time_steps': 10, 'batch_size': 32, 'embedding_size': 200, 'vocab': 1964, 'decay': 0.0}
# Test loss: 0.567963448149
# Test accuracy: 0.873767259911

