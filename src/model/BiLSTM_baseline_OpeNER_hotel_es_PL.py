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


# Train on 3906 samples, validate on 482 samples
# Epoch 1/15
# 4s - loss: 0.4699 - acc: 0.8392 - val_loss: 0.3410 - val_acc: 0.8320
# Epoch 2/15
# 2s - loss: 0.1845 - acc: 0.9391 - val_loss: 0.2937 - val_acc: 0.8651
# Epoch 3/15
# 2s - loss: 0.0914 - acc: 0.9716 - val_loss: 0.3153 - val_acc: 0.8320
# Epoch 4/15
# 2s - loss: 0.0675 - acc: 0.9772 - val_loss: 0.3391 - val_acc: 0.8299
# Epoch 5/15
# 2s - loss: 0.0538 - acc: 0.9857 - val_loss: 0.3820 - val_acc: 0.8299
# Epoch 6/15
# 2s - loss: 0.0484 - acc: 0.9852 - val_loss: 0.3690 - val_acc: 0.8402
# Epoch 7/15
# 2s - loss: 0.0484 - acc: 0.9841 - val_loss: 0.3918 - val_acc: 0.8278
# Epoch 8/15
# 2s - loss: 0.0427 - acc: 0.9869 - val_loss: 0.4189 - val_acc: 0.8257
# Epoch 9/15
# 2s - loss: 0.0392 - acc: 0.9857 - val_loss: 0.3955 - val_acc: 0.8527
# Epoch 10/15
# 2s - loss: 0.0410 - acc: 0.9875 - val_loss: 0.4245 - val_acc: 0.8444
# Epoch 11/15
# 2s - loss: 0.0357 - acc: 0.9864 - val_loss: 0.4232 - val_acc: 0.8568
# Epoch 12/15
# 2s - loss: 0.0310 - acc: 0.9872 - val_loss: 0.4501 - val_acc: 0.8527
# Epoch 13/15
# 2s - loss: 0.0301 - acc: 0.9890 - val_loss: 0.4613 - val_acc: 0.8527
# Epoch 14/15
# 2s - loss: 0.0288 - acc: 0.9887 - val_loss: 0.4878 - val_acc: 0.8527
# Epoch 15/15
# 2s - loss: 0.0268 - acc: 0.9898 - val_loss: 0.5139 - val_acc: 0.8527
# Config: {'drop_out': 0.2, 'decay': 0.0, 'vocab': 2082, 'batch_size': 32, 'LSTM_cell': 64, 'time_steps': 10, 'epochs': 15, 'embedding_size': 200, 'learning_rate': 0.001}
# Test loss: 0.513925915121
# Test accuracy: 0.852697095436

