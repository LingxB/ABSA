from collections import Counter

def freq_dist(tokens):
    c = Counter()
    for t in tokens:
        c.update(t)
    return c


def w_index(counter, start_idx=3):
    w_idx = {w: i + start_idx for i, (w, c) in enumerate(counter.most_common())}
    return w_idx


def df2feats(df, colname, w_idx):
    data = df[colname].apply(lambda x: [w_idx[w] for w in x]).values
    return data


def train_test(data, test_per=0.25):
    split = int(data.shape[0] * (1 - 0.25))

    if len(data.shape) > 1:
        train = data[:split, :]
        test = data[split:, :]
    else:
        train = data[:split]
        test = data[split:]
    return train, test