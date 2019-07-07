import json
import numpy as np
from . import util
import collections

def load(path):
    """
    load the file from jsonline
    :param path:
    :return: examples with many example(dict): {"clusters":[[[mention],[mention]],[another cluster]],
    "doc_key":"str","speakers":[[,,,],[]...],"sentence":[[][]]}
    """
    with open(path) as f:
        train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
        return train_examples

def get_vocab():
    """
    从所有的句子中得到最终的字典,被main调用,不止是train，还有dev和test
    :param examples:
    :return: word2id & id2word
    """
    word2id = {'PAD':0,'UNK':1}
    id2word = {0:'PAD',1:'UNK'}
    index = 2
    data = [load("../data/train.english.jsonlines"),load("../data/dev.english.jsonlines"),load("../data/test.english.jsonlines")]
    for examples in data:
        for example in examples:
            for sent in example["sentences"]:
                for word in sent:
                    if(word not in word2id):
                        word2id[word]=index
                        id2word[index] = word
                        index += 1
    return word2id,id2word

def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    else:
        return v

# 加载glove得到embedding
def get_emb(id2word,embedding_size):
    glove_oov = 0
    turian_oov = 0
    both = 0
    glove_emb_path = "../data/glove.840B.300d.txt.filtered"
    turian_emb_path = "../data/turian.50d.txt"
    word_num = len(id2word)
    emb = np.zeros((word_num,embedding_size))
    glove_emb_dict = util.load_embedding_dict(glove_emb_path,300,"txt")
    turian_emb_dict = util.load_embedding_dict(turian_emb_path,50,"txt")
    for i in range(word_num):
        if id2word[i] in glove_emb_dict:
            word_embedding = glove_emb_dict.get(id2word[i])
            emb[i][0:300] = np.array(word_embedding)
        else:
            # print(id2word[i])
            glove_oov += 1
        if id2word[i] in turian_emb_dict:
            word_embedding = turian_emb_dict.get(id2word[i])
            emb[i][300:350] = np.array(word_embedding)
        else:
            # print(id2word[i])
            turian_oov += 1
        if id2word[i] not in glove_emb_dict and id2word[i] not in turian_emb_dict:
            both += 1
        emb[i] = normalize(emb[i])
    print("embedding num:"+str(word_num))
    print("glove num:"+str(glove_oov))
    print("glove oov rate:"+str(glove_oov/word_num))
    print("turian num:"+str(turian_oov))
    print("turian oov rate:"+str(turian_oov/word_num))
    print("both num:"+str(both))
    return emb


def _doc2vec(doc,word2id,char_dict,max_filter,max_sentences,is_train):
    max_len = 0
    max_word_length = 0
    docvex = []
    length = []
    if is_train:
        sent_num = min(max_sentences,len(doc))
    else:
        sent_num = len(doc)

    for i in range(sent_num):
        sent = doc[i]
        length.append(len(sent))
        if (len(sent) > max_len):
            max_len = len(sent)
        sent_vec =[]
        for j,word in enumerate(sent):
            if len(word)>max_word_length:
                max_word_length = len(word)
            if word in word2id:
                sent_vec.append(word2id[word])
            else:
                sent_vec.append(word2id["UNK"])
        docvex.append(sent_vec)

    char_index = np.zeros((sent_num, max_len, max_word_length),dtype=int)
    for i in range(sent_num):
        sent = doc[i]
        for j,word in enumerate(sent):
            char_index[i, j, :len(word)] = [char_dict[c] for c in word]

    return docvex,char_index,length,max_len

# TODO 修改了接口，确认所有该修改的地方都修改好
def doc2numpy(doc,word2id,chardict,max_filter,max_sentences,is_train):
    docvec, char_index, length, max_len = _doc2vec(doc,word2id,chardict,max_filter,max_sentences,is_train)
    assert max(length) == max_len
    assert char_index.shape[0]==len(length)
    assert char_index.shape[1]==max_len
    doc_np = np.zeros((len(docvec), max_len), int)
    for i in range(len(docvec)):
        for j in range(len(docvec[i])):
            doc_np[i][j] = docvec[i][j]
    return doc_np,char_index,length

# TODO 没有测试
def speaker2numpy(speakers_raw,max_sentences,is_train):
    if is_train and len(speakers_raw)> max_sentences:
        speakers_raw = speakers_raw[0:max_sentences]
    speakers = flatten(speakers_raw)
    speaker_dict = {s: i for i, s in enumerate(set(speakers))}
    speaker_ids = np.array([speaker_dict[s] for s in speakers])
    return speaker_ids


def flat_cluster(clusters):
    flatted = []
    for cluster in clusters:
        for item in cluster:
            flatted.append(item)
    return flatted

def get_right_mention(clusters,mention_start_np,mention_end_np):
    flatted = flat_cluster(clusters)
    cluster_num = len(flatted)
    mention_num = mention_start_np.shape[0]
    right_mention = np.zeros(mention_num,dtype=int)
    for i in range(mention_num):
        if [mention_start_np[i],mention_end_np[i]] in flatted:
            right_mention[i]=1
    return right_mention,cluster_num

def handle_cluster(clusters):
    gold_mentions = sorted(tuple(m) for m in flatten(clusters))
    gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id
    gold_starts, gold_ends = tensorize_mentions(gold_mentions)
    return cluster_ids, gold_starts, gold_ends

# 展平
def flatten(l):
    return [item for sublist in l for item in sublist]

# 把mention分成start end
def tensorize_mentions(mentions):
    if len(mentions) > 0:
        starts, ends = zip(*mentions)
    else:
        starts, ends = [], []
    return np.array(starts), np.array(ends)

def get_char_dict(path):
    vocab = ["<UNK>"]
    with open(path) as f:
        vocab.extend(c.strip() for c in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict

def get_labels(clusters,mention_starts,mention_ends,max_antecedents):
    cluster_ids, gold_starts, gold_ends = handle_cluster(clusters)
    num_mention = mention_starts.shape[0]
    num_gold = gold_starts.shape[0]
    max_antecedents = min(max_antecedents, num_mention)
    mention_indices = {}

    for i in range(num_mention):
        mention_indices[(mention_starts[i].detach().item(), mention_ends[i].detach().item())] = i
    # 用来记录哪些mention是对的，-1表示错误，正数代表这个mention实际上对应哪个gold cluster的id
    mention_cluster_ids = [-1] * num_mention
    # test
    right_mention_count = 0
    for i in range(num_gold):
        right_mention = mention_indices.get((gold_starts[i], gold_ends[i]))
        if (right_mention != None):
            right_mention_count += 1
            mention_cluster_ids[right_mention] = cluster_ids[i]

    # i j 是否属于同一个cluster
    labels = np.zeros((num_mention, max_antecedents + 1), dtype=bool)  # [num_mention,max_an+1]
    for i in range(num_mention):
        ante_count = 0
        null_label = True
        for j in range(max(0, i - max_antecedents), i):
            if (mention_cluster_ids[i] >= 0 and mention_cluster_ids[i] == mention_cluster_ids[j]):
                labels[i, ante_count + 1] = True
                null_label = False
            else:
                labels[i, ante_count + 1] = False
            ante_count += 1
        for j in range(ante_count, max_antecedents):
            labels[i, j + 1] = False
        labels[i, 0] = null_label
    return labels

# test===========================


if __name__=="__main__":
    word2id,id2word = get_vocab()
    get_emb(id2word,350)


