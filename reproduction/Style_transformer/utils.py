import torch

def tensor2text(vocab, tensor):
    tensor = tensor.cpu().numpy()
    text = []
    index2word = vocab.itos
    eos_idx = vocab.stoi['<eos>']
    unk_idx = vocab.stoi['<unk>']
    stop_idxs = [vocab.stoi['!'], vocab.stoi['.'], vocab.stoi['?']]
    for sample in tensor:
        sample_filtered = []
        prev_token = None
        for idx in list(sample):
            if prev_token in stop_idxs:
                break
            if idx == unk_idx or idx == prev_token or idx == eos_idx:
                continue
            prev_token = idx
            sample_filtered.append(index2word[idx])
            
        sample = ' '.join(sample_filtered)
        text.append(sample)

    return text

def calc_ppl(log_probs, tokens_mask):
    return (log_probs.sum() / tokens_mask.sum()).exp()

def idx2onehot(x, num_classes):
    y = x.unsqueeze(-1)
    x_onehot = torch.zeros_like(y.expand(x.size() + torch.Size((num_classes, ))))
    x_onehot.scatter_(-1, y, 1)
    return x_onehot.float()

def word_shuffle(x, l, shuffle_len):
    if not shuffle_len:
        return x

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    pad_mask = (pos_idx >= l.unsqueeze(1)).float()

    scores = pos_idx.float() + ((1 - pad_mask) * noise + pad_mask) * shuffle_len
    x2 = x.clone()
    x2 = x2.gather(1, scores.argsort(1))

    return x2



def word_dropout_raw(x, l, unk_drop_prob, rand_drop_prob, vocab):
    if not unk_drop_prob and not rand_drop_prob:
        return x

    assert unk_drop_prob + rand_drop_prob <= 1

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < l.unsqueeze(1)

    x2 = x.clone()
    
    # drop to <unk> token
    if unk_drop_prob:
        unk_idx = vocab.stoi['<unk>']
        unk_drop_mask = (noise < unk_drop_prob) & token_mask
        x2.masked_fill_(unk_drop_mask, unk_idx)

    # drop to random_mask
    if rand_drop_prob:
        rand_drop_mask = (noise > 1 - rand_drop_prob) & token_mask
        rand_tokens = torch.randint_like(x, len(vocab))
        rand_tokens.masked_fill_(1 - rand_drop_mask, 0)
        x2.masked_fill_(rand_drop_mask, 0)
        x2 = x2 + rand_tokens
    
    return x2

def unk_dropout_(x, l, drop_prob, unk_idx):
    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < l.unsqueeze(1)
    unk_drop_mask = (noise < drop_prob) & token_mask
    x.masked_fill_(unk_drop_mask, unk_idx)

def rand_dropout_(x, l, drop_prob, vocab_size):
    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < l.unsqueeze(1)
    rand_drop_mask = (noise < drop_prob) & token_mask
    rand_tokens = torch.randint_like(x, vocab_size)
    rand_tokens.masked_fill_(1 - rand_drop_mask, 0)
    x.masked_fill_(rand_drop_mask, 0)
    x += rand_tokens

def word_dropout_new(x, l, unk_drop_fac, rand_drop_fac, drop_prob, vocab):
    if not unk_drop_fac and not rand_drop_fac:
        return x

    assert unk_drop_fac + rand_drop_fac <= 1

    batch_size = x.size(0)
    unk_idx = vocab.stoi['<unk>']
    unk_drop_idx = int(batch_size * unk_drop_fac)
    rand_drop_idx = int(batch_size * rand_drop_fac)

    shuffle_idx = torch.argsort(torch.rand(batch_size))
    orignal_idx = torch.argsort(shuffle_idx)

    x2 = x.clone()
    x2 = x2[shuffle_idx]
    
    if unk_drop_idx:
        unk_dropout_(x2[:unk_drop_idx], l[:unk_drop_idx], drop_prob, unk_idx)

    if rand_drop_idx:
        rand_dropout_(x2[-rand_drop_idx:], l[-rand_drop_idx:], drop_prob, len(vocab))

    x2 = x2[orignal_idx]

    return x2

def word_dropout(x, l, drop_prob, unk_idx):
    if not drop_prob:
        return x

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < l.unsqueeze(1)

    drop_mask = (noise < drop_prob) & token_mask
    x2 = x.clone()
    x2.masked_fill_(drop_mask, unk_idx)
    
    return x2

def word_drop(x, l, drop_prob, pad_idx):
    if not drop_prob:
        return x

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < (l.unsqueeze(1) - 1)

    drop_mask = (noise < drop_prob) & token_mask
    x2 = x.clone()
    pos_idx.masked_fill_(drop_mask, x.size(1) - 1)
    pos_idx = torch.sort(pos_idx, 1)[0]
    x2 = x2.gather(1, pos_idx)
    
    return x2

def add_noise(words, lengths, shuffle_len, drop_prob, unk_idx):
    words = word_shuffle(words, lengths, shuffle_len)
    words = word_dropout(words, lengths, drop_prob, unk_idx)
    return words 
