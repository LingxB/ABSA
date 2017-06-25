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


# Train on 3643 samples, validate on 507 samples
# Epoch 1/15
# 20s - loss: 1.0360 - acc: 0.6668 - val_loss: 0.6096 - val_acc: 0.7673
# Epoch 2/15
# 14s - loss: 0.4435 - acc: 0.8696 - val_loss: 0.5101 - val_acc: 0.8343
# Epoch 3/15
# 13s - loss: 0.2452 - acc: 0.9322 - val_loss: 0.5253 - val_acc: 0.8284
# Epoch 4/15
# 13s - loss: 0.1711 - acc: 0.9489 - val_loss: 0.5696 - val_acc: 0.8087
# Epoch 5/15
# 13s - loss: 0.1343 - acc: 0.9607 - val_loss: 0.6287 - val_acc: 0.8087
# Epoch 6/15
# 13s - loss: 0.1185 - acc: 0.9621 - val_loss: 0.6534 - val_acc: 0.8047
# Epoch 7/15
# 14s - loss: 0.0985 - acc: 0.9651 - val_loss: 0.7063 - val_acc: 0.8028
# Epoch 8/15
# 14s - loss: 0.0950 - acc: 0.9698 - val_loss: 0.7467 - val_acc: 0.7988
# Epoch 9/15
# 13s - loss: 0.0814 - acc: 0.9709 - val_loss: 0.7649 - val_acc: 0.8028
# Epoch 10/15
# 13s - loss: 0.0803 - acc: 0.9731 - val_loss: 0.7738 - val_acc: 0.8028
# Epoch 11/15
# 14s - loss: 0.0769 - acc: 0.9734 - val_loss: 0.8084 - val_acc: 0.8107
# Epoch 12/15
# 14s - loss: 0.0708 - acc: 0.9731 - val_loss: 0.8101 - val_acc: 0.8067
# Epoch 13/15
# 14s - loss: 0.0681 - acc: 0.9736 - val_loss: 0.8644 - val_acc: 0.7988
# Epoch 14/15
# 14s - loss: 0.0652 - acc: 0.9758 - val_loss: 0.8831 - val_acc: 0.7968
# Epoch 15/15
# 14s - loss: 0.0608 - acc: 0.9767 - val_loss: 0.8779 - val_acc: 0.8067
# Config: {'epochs': 15, 'vocab': 1964, 'decay': 0.0, 'drop_out': 0.2, 'time_steps': 10, 'LSTM_cell': 64, 'embedding_size': 200, 'batch_size': 32, 'learning_rate': 0.001}
# Test loss: 0.877918887421
# Test accuracy: 0.806706114281


