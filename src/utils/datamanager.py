
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

def get_root(file):
    root = ET.parse(file).getroot()
    return root

def get_text(root):
    text = [e.text for c in root.getchildren() for e in c.findall('wf') if c.tag == 'text']
    return text

def get_term(root):
    """return map between tid and wid"""
    terms = [e for c in root.getchildren() for e in c.findall('term') if c.tag == 'terms']
    tw_map = {x.findall('span')[0].findall('target')[0].attrib['id']: x.attrib['tid'] for x in terms}
    return tw_map

def get_opinions(root):
    """Return target, expression, polarity"""
    opinions = [e for c in root.getchildren() for e in c.findall('opinion') if c.tag == 'opinions']
    triples = []
    for opinion in opinions:
        try:
            targets = [o.findall('span')[0].findall('target') for o in opinion.getchildren() if o.tag == 'opinion_target'][0]
            t_id = [t.attrib['id'] for t in targets]
        except IndexError:
            t_id = [None]
        exps = [e.findall('span')[0].findall('target') for e in opinion.getchildren() if e.tag == 'opinion_expression'][0]
        e_id = [e.attrib['id'] for e in exps]
        polarity = [e.attrib['polarity'] for e in opinion.getchildren() if e.tag == 'opinion_expression'][0]
        triples.append((t_id,e_id,polarity))
    return triples

def id2words(opinions, root):
    text = get_text(root)
    w_opins = []
    for o in opinions:
        target = [w for w in map(lambda x: text[int(x.split('t')[-1])-1] if x else None,o[0])]
        exp = [w for w in map(lambda x: text[int(x.split('t')[-1])-1],o[1])]
        polarity = o[2]
        if target[0]:
            w_opins.append([' '.join(target),' '.join(exp),polarity])
        else:
            w_opins.append([None,' '.join(exp),polarity])
    return np.array(w_opins)

def extract_data(file, colnames=['TARGET', 'OEXP', 'POLARITY']):
    print(file)
    root = get_root(file)
    opinions_id = get_opinions(root)
    opinions = id2words(opinions_id, root)
    df = pd.DataFrame(opinions, columns=colnames)
    return df

def iter_data(file, colnames=['TARGET', 'OEXP', 'POLARITY']):
    # print(file)
    root = get_root(file)
    opinions_id = get_opinions(root)
    if opinions_id:
        opinions = id2words(opinions_id, root)
        df = pd.DataFrame(opinions, columns=colnames)
        yield df
    else:
        pass

def atae_converter(f_path):
    with open(f_path,'r',encoding='utf-8') as f:
        temp = 0
        temp_lst = ['SENT','ASP','CLS']
        d = {t:[] for t in temp_lst}
        for line in f:
            if temp == 0:
                d[temp_lst[0]].append(line.strip())
                temp = 1
            elif temp ==1:
                d[temp_lst[1]].append(line.strip())
                temp = 2
            elif temp ==2:
                d[temp_lst[2]].append(int(line.strip()))
                temp = 0
    return pd.DataFrame(data=d)