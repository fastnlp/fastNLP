import pickle

import Word2Idx


def get_sets(m, n):
    """
    get a train set containing m samples and a test set containing n samples
    """
    samples = pickle.load(open("tuples.pkl","rb"))
    if m+n > len(samples):
        print("asking for too many tuples\n")
        return
    train_samples = samples[ : m]
    test_samples = samples[m: m+n]
    return train_samples, test_samples

def build_wordidx():
    """
    build wordidx using word2idx
    """
    train, test = get_sets(500000, 2000)
    words = []
    for x in train:
        words += x[0]
    wordidx = Word2Idx.Word2Idx()
    wordidx.build(words)
    print(wordidx.num)
    print(wordidx.i2w(0))
    wordidx.save("wordidx.pkl")

def build_sets():
    """
    build train set and test set, transform word to index
    """
    train, test = get_sets(500000, 2000)
    wordidx = Word2Idx.Word2Idx()
    wordidx.load("wordidx.pkl")
    train_set = []
    for x in train:
        sent = [wordidx.w2i(w) for w in x[0]]
        train_set.append({"sent" : sent, "class" : x[1]})
    test_set = []
    for x in test:
        sent = [wordidx.w2i(w) for w in x[0]]
        test_set.append({"sent" : sent, "class" : x[1]})
    pickle.dump(train_set, open("train_set.pkl", "wb"))
    pickle.dump(test_set, open("test_set.pkl", "wb"))

if __name__ == "__main__":
    build_wordidx()
    build_sets()
