from os import walk
import pandas as pd
import src.utils.datamanager as dm

path_en = 'Data\OpeNER\opinion_annotations_en\kaf\hotel/'
path_es = 'Data\OpeNER\opinion_annotations_es\kaf\hotel/'
out_path = 'Data\OpeNER/'

files_en = [filename for (dirpath, dirnames, filename) in walk(path_en)][0]
files_es = [filename for (dirpath, dirnames, filename) in walk(path_es)][0]

#########################
# Baseline dataset
#########################

df_en = pd.concat([df for file in files_en for df in dm.iter_data(path_en+file) ], axis=0, ignore_index=True)
df_es = pd.concat([df for file in files_es for df in dm.iter_data(path_es+file) ], axis=0, ignore_index=True)

df_en.to_csv(out_path+'OpeNER_hotel_en.csv', index=False, encoding='utf-8')
df_es.to_csv(out_path+'OpeNER_hotel_es.csv', index=False, encoding='utf-8')


#########################
# Patrick's Train/Test
#########################

def read_fid(file):
    """Read file ids to list"""
    with open(file, 'r', encoding='utf-8') as f:
        l = [line.strip()+'.kaf' for line in f]
    return l

def id2df(fids, path):
    df = pd.concat([df for file in fids for df in dm.iter_data(path + file)], axis=0, ignore_index=True)
    return df

train_id_en = read_fid('Data/OpeNER/PL/opener-hotel-en/ids.train')
test_id_en = read_fid('Data/OpeNER/PL/opener-hotel-en/ids.test')

train_id_es = read_fid('Data/OpeNER/PL/opener-hotel-es/ids.train')
test_id_es = read_fid('Data/OpeNER/PL/opener-hotel-es/ids.test')

df_en_train = id2df(train_id_en,path_en)
df_en_test = id2df(test_id_en,path_en)

df_es_train = id2df(train_id_es,path_es)
df_es_test = id2df(test_id_es,path_es)

df_en_train.to_csv(out_path+'OpeNER_hotel_en_train.csv', index=False, encoding='utf-8')
df_en_test.to_csv(out_path+'OpeNER_hotel_en_test.csv', index=False, encoding='utf-8')
df_es_train.to_csv(out_path+'OpeNER_hotel_es_train.csv', index=False, encoding='utf-8')
df_es_test.to_csv(out_path+'OpeNER_hotel_es_test.csv', index=False, encoding='utf-8')


#########################
# atae-LSTM styple
#########################
df_en_train = pd.read_csv('Data/OpeNER/PL/OpeNER_hotel_en_train.csv')
df_en_test = pd.read_csv('Data/OpeNER/PL/OpeNER_hotel_en_test.csv')
df_es_train = pd.read_csv('Data/OpeNER/PL/OpeNER_hotel_es_train.csv')
df_es_test = pd.read_csv('Data/OpeNER/PL/OpeNER_hotel_es_test.csv')


def atae_style(df, path):
    _df = df.copy()
    _df['CLASS'] = ~_df.POLARITY.str.contains('Negative')
    with open(path,'w',encoding='utf-8') as f:
        for idx,row in _df.iterrows():
                exp = row.OEXP
                target = row.TARGET
                polar = 1 if row.CLASS else -1
                try:
                    f.write(exp+'\n'+str(target)+'\n'+str(polar)+'\n')
                except TypeError:
                    print(exp,target,polar)

atae_style(df_en_train, 'Data/OpeNER/PL/OpeNER_hotel_en_train.cor')
atae_style(df_en_test, 'Data/OpeNER/PL/OpeNER_hotel_en_test.cor')
atae_style(df_es_train, 'Data/OpeNER/PL/OpeNER_hotel_es_train.cor')
atae_style(df_es_test, 'Data/OpeNER/PL/OpeNER_hotel_es_test.cor')

















