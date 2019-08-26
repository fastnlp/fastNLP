import torch
import torch.nn as nn
from fastNLP.core import Const as C


class DPCNN(nn.Module):
    def __init__(self, init_embed, num_cls, n_filters=256,
                 kernel_size=3, n_layers=7, embed_dropout=0.1, cls_dropout=0.1):
        super().__init__()
        self.region_embed = RegionEmbedding(
            init_embed, out_dim=n_filters, kernel_sizes=[1, 3, 5])
        embed_dim = self.region_embed.embedding_dim
        self.conv_list = nn.ModuleList()
        for i in range(n_layers):
            self.conv_list.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size,
                          padding=kernel_size//2),
                nn.Conv1d(n_filters, n_filters, kernel_size,
                          padding=kernel_size//2),
            ))
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.embed_drop = nn.Dropout(embed_dropout)
        self.classfier = nn.Sequential(
            nn.Dropout(cls_dropout),
            nn.Linear(n_filters, num_cls),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=0.01)

    def forward(self, words, seq_len=None):
        words = words.long()
        # get region embeddings
        x = self.region_embed(words)
        x = self.embed_drop(x)

        # not pooling on first conv
        x = self.conv_list[0](x) + x
        for conv in self.conv_list[1:]:
            x = self.pool(x)
            x = conv(x) + x

        # B, C, L => B, C
        x, _ = torch.max(x, dim=2)
        x = self.classfier(x)
        return {C.OUTPUT: x}

    def predict(self, words, seq_len=None):
        x = self.forward(words, seq_len)[C.OUTPUT]
        return {C.OUTPUT: torch.argmax(x, 1)}


class RegionEmbedding(nn.Module):
    def __init__(self, init_embed, out_dim=300, kernel_sizes=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [5, 9]
        assert isinstance(
            kernel_sizes, list), 'kernel_sizes should be List(int)'
        # self.embed = nn.Embedding.from_pretrained(torch.tensor(init_embed).float(), freeze=False)
        self.embed = init_embed
        try:
            embed_dim = self.embed.embedding_dim
        except Exception:
            embed_dim = self.embed.embed_size
        self.region_embeds = nn.ModuleList()
        for ksz in kernel_sizes:
            self.region_embeds.append(nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, ksz, padding=ksz // 2),
            ))
        self.linears = nn.ModuleList([nn.Conv1d(embed_dim, out_dim, 1)
                                      for _ in range(len(kernel_sizes))])
        self.embedding_dim = embed_dim

    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(1, 2)
        # B, C, L
        out = 0
        for conv, fc in zip(self.region_embeds, self.linears[1:]):
            conv_i = conv(x)
            out = out + fc(conv_i)
        # B, C, L
        return out


if __name__ == '__main__':
    x = torch.randint(0, 10000, size=(5, 15), dtype=torch.long)
    model = DPCNN((10000, 300), 20)
    y = model(x)
    print(y.size(), y.mean(1), y.std(1))
