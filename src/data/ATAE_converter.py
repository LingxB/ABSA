from src.utils.datamanager import atae_converter


train = atae_converter('F:\PhD\ABSA\AT-LSTM/atae-lstm\data/train.cor')
test = atae_converter('F:\PhD\ABSA\AT-LSTM/atae-lstm\data/test.cor')
dev = atae_converter('F:\PhD\ABSA\AT-LSTM/atae-lstm\data\dev.cor')

train.to_csv('F:\PhD\ABSA\Data\ATAE-LSTM/train.csv',index=False)
test.to_csv('F:\PhD\ABSA\Data\ATAE-LSTM/test.csv',index=False)
dev.to_csv('F:\PhD\ABSA\Data\ATAE-LSTM/dev.csv',index=False)

