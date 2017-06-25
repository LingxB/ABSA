from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.optimizers import Adam
from keras import backend as K
import pandas as pd
import numpy as np
from src.utils.preprocess import *


config = {'embedding_size': 200,
          'time_steps': 25,
          'drop_out': 0.2,
          'LSTM_cell': 64,
          'learning_rate': 0.001,
          'decay': 0.0,
          'batch_size': 32,
          'epochs': 25
         }

train_path = 'F:\PhD\ABSA\Data\ATAE-LSTM/train.csv'
test_path = 'F:\PhD\ABSA\Data\ATAE-LSTM/test.csv'
dev_path = 'F:\PhD\ABSA\Data\ATAE-LSTM\dev.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
dev = pd.read_csv(dev_path)

dataset = pd.concat([train,test,dev])



# Preprocessing
def global_stats(df):
    df['TOKENS'] = df.SENT.apply(lambda x: x.split())
    df['TLEN'] = df.TOKENS.apply(lambda x: len(x))
    #df['CLASS'] = ~df.POLARITY.str.contains('Negative')
    c = freq_dist(df.TOKENS)
    w_idx = w_index(c, start_idx=1)
    config.update({'vocab':len(w_idx)})
    return w_idx

def input_ready(df, mlen, w_idx):
    df['TOKENS'] = df.SENT.apply(lambda x: x.split())
    df['TLEN'] = df.TOKENS.apply(lambda x: len(x))
    #df['CLASS'] = ~df.POLARITY.str.contains('Negative')

    data = df2feats(df, 'TOKENS', w_idx)
    X = sequence.pad_sequences(data, maxlen=mlen).astype('float32')
    y = pd.get_dummies(df.CLS).values.astype('float32')
    return X,y

w_idx = global_stats(dataset)
X_train,y_train = input_ready(train, config['time_steps'], w_idx)
X_test,y_test = input_ready(test, config['time_steps'], w_idx)
X_dev,y_dev = input_ready(dev, config['time_steps'], w_idx)


# data,labels = input_ready(dataset, config['time_steps'])
# X_train,X_test = train_test(data)
# y_train,y_test = train_test(labels)

adam = Adam(lr=config['learning_rate'], decay=config['decay'])
model = Sequential()
model.add(Embedding(config['vocab']+2, config['embedding_size'], input_length=config['time_steps'], mask_zero=True))
model.add(Bidirectional(LSTM(config['LSTM_cell'], dropout=config['drop_out'], recurrent_dropout=config['drop_out'])))
#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(config['drop_out']))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train,
          batch_size=config['batch_size'],
          epochs=config['epochs'],
          verbose=2,
          validation_data=(X_dev, y_dev))

loss, acc = model.evaluate(X_test, y_test,
                           batch_size=config['batch_size'],
                           verbose=2)

print('\nConfig:', config)
print('Test loss:', loss)
print('Test accuracy:', acc)



prediction = model.predict(X_test)
prediction = np.apply_along_axis(np.argmax, 1, prediction)
labels = np.apply_along_axis(np.argmax, 1, pd.get_dummies(test.CLS).values)
val_matrix = pd.DataFrame({'PREDS':prediction,'LABELS':labels})
val_matrix = val_matrix.ix[~(val_matrix.LABELS==1)]
val_matrix['CROSS'] = val_matrix.LABELS==val_matrix.PREDS
print('Binary test acc:', val_matrix.CROSS.value_counts()[True]/val_matrix.shape[0])

# Train on 2990 samples, validate on 528 samples
# Epoch 1/25
# 38s - loss: 0.8505 - acc: 0.6301 - val_loss: 0.7028 - val_acc: 0.6837
# Epoch 2/25
# 31s - loss: 0.5787 - acc: 0.7522 - val_loss: 0.6152 - val_acc: 0.7273
# Epoch 3/25
# 30s - loss: 0.3887 - acc: 0.8472 - val_loss: 0.6185 - val_acc: 0.7386
# Epoch 4/25
# 30s - loss: 0.2659 - acc: 0.9013 - val_loss: 0.6841 - val_acc: 0.7386
# Epoch 5/25
# 31s - loss: 0.2043 - acc: 0.9301 - val_loss: 0.7162 - val_acc: 0.7330
# Epoch 6/25
# 31s - loss: 0.1718 - acc: 0.9388 - val_loss: 0.8208 - val_acc: 0.7386
# Epoch 7/25
# 30s - loss: 0.1480 - acc: 0.9431 - val_loss: 0.8297 - val_acc: 0.7386
# Epoch 8/25
# 30s - loss: 0.1342 - acc: 0.9482 - val_loss: 0.8192 - val_acc: 0.7405
# Epoch 9/25
# 32s - loss: 0.1232 - acc: 0.9552 - val_loss: 0.8621 - val_acc: 0.7330
# Epoch 10/25
# 30s - loss: 0.1162 - acc: 0.9512 - val_loss: 0.9126 - val_acc: 0.7197
# Epoch 11/25
# 30s - loss: 0.1191 - acc: 0.9512 - val_loss: 0.9379 - val_acc: 0.7292
# Epoch 12/25
# 31s - loss: 0.1057 - acc: 0.9528 - val_loss: 0.9788 - val_acc: 0.7424
# Epoch 13/25
# 31s - loss: 0.1037 - acc: 0.9538 - val_loss: 0.9452 - val_acc: 0.7348
# Epoch 14/25
# 32s - loss: 0.1044 - acc: 0.9545 - val_loss: 0.9526 - val_acc: 0.7405
# Epoch 15/25
# 31s - loss: 0.0959 - acc: 0.9565 - val_loss: 1.0811 - val_acc: 0.7330
# Epoch 16/25
# 30s - loss: 0.0901 - acc: 0.9589 - val_loss: 1.0699 - val_acc: 0.7273
# Epoch 17/25
# 29s - loss: 0.0887 - acc: 0.9572 - val_loss: 1.0689 - val_acc: 0.7292
# Epoch 18/25
# 32s - loss: 0.0882 - acc: 0.9579 - val_loss: 1.0242 - val_acc: 0.7348
# Epoch 19/25
# 32s - loss: 0.0851 - acc: 0.9572 - val_loss: 1.0607 - val_acc: 0.7348
# Epoch 20/25
# 31s - loss: 0.0843 - acc: 0.9565 - val_loss: 1.2016 - val_acc: 0.7462
# Epoch 21/25
# 31s - loss: 0.0825 - acc: 0.9535 - val_loss: 1.1500 - val_acc: 0.7443
# Epoch 22/25
# 30s - loss: 0.0789 - acc: 0.9575 - val_loss: 1.1620 - val_acc: 0.7348
# Epoch 23/25
# 31s - loss: 0.0760 - acc: 0.9562 - val_loss: 1.1397 - val_acc: 0.7292
# Epoch 24/25
# 30s - loss: 0.0809 - acc: 0.9585 - val_loss: 1.1646 - val_acc: 0.7367
# Epoch 25/25
# 30s - loss: 0.0762 - acc: 0.9542 - val_loss: 1.1486 - val_acc: 0.7386
# Config: {'embedding_size': 200, 'drop_out': 0.2, 'learning_rate': 0.001, 'vocab': 5175, 'epochs': 25, 'batch_size': 32, 'time_steps': 25, 'LSTM_cell': 64, 'decay': 0.0}
# Test loss: 1.10617020037
# Test accuracy: 0.734840699176
# Binary test acc: 0.765642775882



