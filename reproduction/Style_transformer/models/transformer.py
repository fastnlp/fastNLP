import math
import torch
from torch import nn
import torch.nn.functional as F
from utils import idx2onehot


class StyleTransformer(nn.Module):
    def __init__(self, config, vocab):
        super(StyleTransformer, self).__init__()
        num_styles, num_layers = config.num_styles, config.num_layers
        d_model, max_length = config.d_model, config.max_length
        h, dropout = config.h, config.dropout
        learned_pos_embed = config.learned_pos_embed
        load_pretrained_embed = config.load_pretrained_embed
        
        self.max_length = config.max_length
        self.eos_idx = vocab.stoi['<eos>']
        self.pad_idx = vocab.stoi['<pad>']
        self.style_embed = Embedding(num_styles, d_model)
        self.embed = EmbeddingLayer(
            vocab, d_model, max_length,
            self.pad_idx,
            learned_pos_embed,
            load_pretrained_embed,
        )
        self.sos_token = nn.Parameter(torch.randn(d_model))
        self.encoder = Encoder(num_layers, d_model, len(vocab), h, dropout)
        self.decoder = Decoder(num_layers, d_model, len(vocab), h, dropout)
        
    def forward(self, inp_tokens, gold_tokens, inp_lengths, style,
                generate=False, differentiable_decode=False, temperature=1.0):
        batch_size = inp_tokens.size(0)
        max_enc_len = inp_tokens.size(1)

        assert max_enc_len <= self.max_length

        pos_idx = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(inp_lengths.device)

        src_mask = pos_idx[:, :max_enc_len] >= inp_lengths.unsqueeze(-1)
        src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), 1)
        src_mask = src_mask.view(batch_size, 1, 1, max_enc_len + 1)

        tgt_mask = torch.ones((self.max_length, self.max_length)).to(src_mask.device)
        tgt_mask = (tgt_mask.tril() == 0).view(1, 1, self.max_length, self.max_length)

        style_emb = self.style_embed(style).unsqueeze(1)

        enc_input = torch.cat((style_emb, self.embed(inp_tokens, pos_idx[:, :max_enc_len])), 1)
        memory = self.encoder(enc_input, src_mask)
        
        sos_token = self.sos_token.view(1, 1, -1).expand(batch_size, -1, -1)
        
        if not generate:
            dec_input = gold_tokens[:, :-1]
            max_dec_len = gold_tokens.size(1)
            dec_input_emb = torch.cat((sos_token, self.embed(dec_input, pos_idx[:, :max_dec_len - 1])), 1)
            log_probs = self.decoder(
                dec_input_emb, memory,
                src_mask, tgt_mask[:, :, :max_dec_len, :max_dec_len],
                temperature
            )
        else:
            
            log_probs = []
            next_token = sos_token
            prev_states = None
            
            for k in range(self.max_length):
                log_prob, prev_states = self.decoder.incremental_forward(
                    next_token, memory,
                    src_mask, tgt_mask[:, :, k:k+1, :k+1],
                    temperature,
                    prev_states
                )

                log_probs.append(log_prob)
                
                if differentiable_decode:
                    next_token = self.embed(log_prob.exp(), pos_idx[:, k:k+1])
                else:
                    next_token = self.embed(log_prob.argmax(-1), pos_idx[:, k:k+1])

                #if (pred_tokens == self.eos_idx).max(-1)[0].min(-1)[0].item() == 1:
                #    break

            log_probs = torch.cat(log_probs, 1)
            
            
        return log_probs
    
class Discriminator(nn.Module):
    def __init__(self, config, vocab):
        super(Discriminator, self).__init__()
        num_styles, num_layers = config.num_styles, config.num_layers
        d_model, max_length = config.d_model, config.max_length
        h, dropout = config.h, config.dropout
        learned_pos_embed = config.learned_pos_embed
        load_pretrained_embed = config.load_pretrained_embed
        num_classes = config.num_classes
        
        self.pad_idx = vocab.stoi['<pad>']
        self.style_embed = Embedding(num_styles, d_model)
        self.embed = EmbeddingLayer(
            vocab, d_model, max_length,
            self.pad_idx,
            learned_pos_embed,
            load_pretrained_embed
        )
        self.cls_token = nn.Parameter(torch.randn(d_model))
        self.encoder = Encoder(num_layers, d_model, len(vocab), h, dropout)
        self.classifier = Linear(d_model, num_classes)
    
    def forward(self, inp_tokens, inp_lengths, style=None):
        batch_size = inp_tokens.size(0)
        num_extra_token = 1 if style is None else 2
        max_seq_len = inp_tokens.size(1)

        pos_idx = torch.arange(max_seq_len).unsqueeze(0).expand((batch_size, -1)).to(inp_lengths.device)
        mask = pos_idx >= inp_lengths.unsqueeze(-1)
        for _ in range(num_extra_token):
            mask = torch.cat((torch.zeros_like(mask[:, :1]), mask), 1)
        mask = mask.view(batch_size, 1, 1, max_seq_len + num_extra_token)

        cls_token = self.cls_token.view(1, 1, -1).expand(batch_size, -1, -1)

        enc_input = cls_token
        if style is not None:
            style_emb = self.style_embed(style).unsqueeze(1)
            enc_input = torch.cat((enc_input, style_emb), 1)
            
        enc_input = torch.cat((enc_input, self.embed(inp_tokens, pos_idx)), 1)
        
        encoded_features = self.encoder(enc_input, mask)
        logits = self.classifier(encoded_features[:, 0])

        return F.log_softmax(logits, -1)
        
        
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, vocab_size, h, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, mask):
        y = x

        assert y.size(1) == mask.size(-1)
            
        for layer in self.layers:
            y = layer(y, mask)

        return self.norm(y)
        
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, vocab_size, h, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
        self.generator = Generator(d_model, vocab_size)

    def forward(self, x, memory, src_mask, tgt_mask, temperature):
        y = x

        assert y.size(1) == tgt_mask.size(-1)
        
        for layer in self.layers:
            y = layer(y, memory, src_mask, tgt_mask)

        return self.generator(self.norm(y), temperature)
    def incremental_forward(self, x, memory, src_mask, tgt_mask, temperature, prev_states=None):
        y = x

        new_states = []

                                            
        for i, layer in enumerate(self.layers):
            y, new_sub_states = layer.incremental_forward(
                y, memory, src_mask, tgt_mask,
                prev_states[i] if prev_states else None
            )

            new_states.append(new_sub_states)
        
        new_states.append(torch.cat((prev_states[-1], y), 1) if prev_states else y)
        y = self.norm(new_states[-1])[:, -1:]
        
        return self.generator(y, temperature), new_states
    
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, temperature):
        return F.log_softmax(self.proj(x) / temperature, dim=-1)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab, d_model, max_length, pad_idx, learned_pos_embed, load_pretrained_embed):
        super(EmbeddingLayer, self).__init__()
        self.token_embed = Embedding(len(vocab), d_model)
        self.pos_embed = Embedding(max_length, d_model)
        self.vocab_size = len(vocab)
        if load_pretrained_embed:
            self.token_embed = nn.Embedding.from_pretrained(vocab.vectors)
            print('embed loaded.')
    def forward(self, x, pos):
        if len(x.size()) == 2:
            y = self.token_embed(x) + self.pos_embed(pos)
        else:
            y = torch.matmul(x, self.token_embed.weight) + self.pos_embed(pos)

        return y
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer =  nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.pw_ffn)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.src_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.pw_ffn)

    def incremental_forward(self, x, memory, src_mask, tgt_mask, prev_states=None):
        new_states = []
        m = memory

        x = torch.cat((prev_states[0], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[0].incremental_forward(x, lambda x: self.self_attn(x[:, -1:], x, x, tgt_mask))
        x = torch.cat((prev_states[1], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[1].incremental_forward(x, lambda x: self.src_attn(x[:, -1:], m, m, src_mask))
        x = torch.cat((prev_states[2], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[2].incremental_forward(x, lambda x: self.pw_ffn(x[:, -1:]))
        return x, new_states  
   
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.head_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x, l in zip((query, key, value), self.head_projs)]

        attn_feature, _ = scaled_attention(query, key, value, mask)

        attn_concated = attn_feature.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.fc(attn_concated)

def scaled_attention(query, key, value, mask):
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    scores.masked_fill_(mask, float('-inf'))
    attn_weight = F.softmax(scores, -1)
    attn_feature = attn_weight.matmul(value)

    return attn_feature, attn_weight
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            Linear(4 * d_model, d_model),
        )
        
    def forward(self, x):
        return self.mlp(x)

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)

    def incremental_forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)
    
def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps)
    return m
