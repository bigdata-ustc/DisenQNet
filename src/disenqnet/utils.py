# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

def get_mask(seq_len, lengths):
    """
        Get pad mask
        :param seq_len: int, padded length of data tensor
        :param lengths: Tensor of long (batch_size), valid length of each data tensor
        :returns: Tensor of bool (batch_size, seq_len), False means data, and True means pad
            [F F F T T T]
    """
    device = lengths.device
    # batch_size
    batch_size = lengths.size(0)
    # seq_len
    pos_index = torch.arange(seq_len).to(device)
    # batch_size * seq_len
    mask = pos_index.unsqueeze(0).expand(batch_size, -1) >= lengths.unsqueeze(-1)
    return mask

def shuffle(real):
    """
        shuffle data in a batch
        [1, 2, 3, 4, 5] -> [2, 3, 4, 5, 1]
        P(X,Y) -> P(X)P(Y) by shuffle Y in a batch
        P(X,Y) = [(1,1'),(2,2'),(3,3')] -> P(X)P(Y) = [(1,2'),(2,3'),(3,1')]
        :param real: Tensor of (batch_size, ...), data, batch_size > 1
        :returns: Tensor of (batch_size, ...), shuffled data
    """
    # |0 1 2 3| => |1 2 3 0|
    device = real.device
    batch_size = real.size(0)
    shuffled_index = (torch.arange(batch_size) + 1) % batch_size
    shuffled_index = shuffled_index.to(device)
    shuffled = real.index_select(dim=0, index=shuffled_index)
    return shuffled

def spectral_norm(W, n_iteration=5):
    """
        Spectral normalization for Lipschitz constrain in Disc of WGAN
        Following https://blog.csdn.net/qq_16568205/article/details/99586056
        |W|^2 = principal eigenvalue of W^TW through power iteration
        v = W^Tu/|W^Tu|
        u = Wv / |Wv|
        |W|^2 = u^TWv
        
        :param w: Tensor of (out_dim, in_dim) or (out_dim), weight matrix of NN
        :param n_iteration: int, number of iterations for iterative calculation of spectral normalization:
        :returns: Tensor of (), spectral normalization of weight matrix
    """
    device = W.device
    # (o, i)
    # bias: (O) -> (o, 1)
    if W.dim() == 1:
        W = W.unsqueeze(-1)
    out_dim, in_dim = W.size()
    # (i, o)
    Wt = W.transpose(0, 1)
    # (1, i)
    u = torch.ones(1, in_dim).to(device)
    for _ in range(n_iteration):
        # (1, i) * (i, o) -> (1, o)
        v = torch.mm(u, Wt)
        v = v / v.norm(p=2)
        # (1, o) * (o, i) -> (1, i)
        u = torch.mm(v, W)
        u = u / u.norm(p=2)
    # (1, i) * (i, o) * (o, 1) -> (1, 1)
    sn = torch.mm(torch.mm(u, Wt), v.transpose(0, 1)).sum() ** 0.5
    return sn

class MLP(nn.Module):
    """
        Multi-Layer Perceptron
        :param in_dim: int, size of input feature
        :param n_classes: int, number of output classes
        :param hidden_dim: int, size of hidden vector
        :param dropout: float, dropout rate
        :param n_layers: int, number of layers, at least 2, default = 2
        :param act: function, activation function, default = leaky_relu
    """

    def __init__(self, in_dim, n_classes, hidden_dim, dropout, n_layers=2, act=F.leaky_relu):
        super(MLP, self).__init__()
        self.l_in = nn.Linear(in_dim, hidden_dim)
        self.l_hs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2))
        self.l_out = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        return
    
    def forward(self, input):
        """
            :param input: Tensor of (batch_size, in_dim), input feature
            :returns: Tensor of (batch_size, n_classes), output class
        """
        hidden = self.act(self.l_in(self.dropout(input)))
        for l_h in self.l_hs:
            hidden = self.act(l_h(self.dropout(hidden)))
        output = self.l_out(self.dropout(hidden))
        return output

class TextCNN(nn.Module):
    """
        TextCNN
        :param embed_dim: int, size of word embedding
        :param hidden_dim: int, size of question embedding
    """

    def __init__(self, embed_dim, hidden_dim):
        super(TextCNN, self).__init__()
        kernel_sizes = [2, 3, 4, 5]
        channel_dim = hidden_dim // len(kernel_sizes)
        self.min_seq_len = max(kernel_sizes)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, channel_dim, k_size) for k_size in kernel_sizes])
        return
    
    def forward(self, embed):
        """
            :param embed: Tensor of (batch_size, seq_len, embed_dim), word embedding
            :returns: Tensor of (batch_size, hidden_dim), question embedding
        """
        if embed.size(1) < self.min_seq_len:
            device = embed.device
            pad = torch.zeros(embed.size(0), self.min_seq_len-embed.size(1), embed.size(-1)).to(device)
            embed = torch.cat((embed, pad), dim=1)
        # (b, s, d) => (b, d, s) => (b, d', s') => (b, d', 1) => (b, d')
        # batch_size * dim * seq_len
        hidden = [F.leaky_relu(conv(embed.transpose(1, 2))) for conv in self.convs]
        # batch_size * dim
        hidden = [F.max_pool1d(h, kernel_size=h.size(2)).squeeze(-1) for h in hidden]
        hidden = torch.cat(hidden, dim=-1)
        return hidden

class Disc(nn.Module):
    """
        2-layer discriminator for MI estimator
        :param x_dim: int, size of x vector
        :param y_dim: int, size of y vector
        :param dropout: float, dropout rate
    """
    def __init__(self, x_dim, y_dim, dropout):
        super(Disc, self).__init__()
        self.disc = MLP(x_dim+y_dim, 1, y_dim, dropout, n_layers=2)
        return
    
    def forward(self, x, y):
        """
            :param x: Tensor of (batch_size, hidden_dim), x
            :param y: Tensor of (batch_size, hidden_dim), y
            :returns: Tensor of (batch_size), score
        """
        input = torch.cat((x, y), dim=-1)
        # (b, 1) -> (b)
        score = self.disc(input).squeeze(-1)
        return score

class MI(nn.Module): 
    """
        MI JS estimator
        MI(X,Y) = E_pxy[-sp(-T(x,y))] - E_pxpy[sp(T(x,y))]
        
        :param x_dim: int, size of x vector
        :param y_dim: int, size of y vector
        :param dropout: float, dropout rate of discrimanator
    """

    def __init__(self, x_dim, y_dim, dropout):
        super(MI, self).__init__()
        self.disc = Disc(x_dim, y_dim, dropout)
        return
    
    def forward(self, x, y):
        """
            :param x: Tensor of (batch_size, hidden_dim), x
            :param y: Tensor of (batch_size, hidden_dim), y
            :returns: Tensor of (), MI
        """
        # P(X,Y) = (x, y), P(X)P(Y) = (x, sy)
        sy = shuffle(y)
        mi = -F.softplus(-self.disc(x, y)).mean() - F.softplus(self.disc(x, sy)).mean()
        return mi

class PairwiseLoss(nn.Module):
    """
        Pairwise rank loss
        L(pos, neg) = max(0, mu-pos+neg) + l2_norm

        :param mu: int, margin
        :param w_norm: float, weight of L2 normalization
        :param reduction: str, mean or sum, reduction type for loss
    """

    def __init__(self, mu, w_norm, reduction="mean"):
        super().__init__()
        self.mu = mu
        self.w_norm = w_norm
        self.reduction = reduction
        return
    
    def forward(self, pos, neg, parameters):
        """
            :param pos: Tensor of (batch_size), positive samples
            :param neg: Tensor of (batch_size), negative samples
            :param parameters: iterable, model parameters
        """
        task_loss = F.relu(self.mu - pos + neg)
        if self.reduction == "mean":
            task_loss = task_loss.mean()
        elif self.reduction == "sum":
            task_loss = task_loss.sum()
        loss = task_loss
        if self.w_norm > 0:
            l2_loss = sum(p.norm(p=2)**2 for p in parameters)
            loss = task_loss + self.w_norm * l2_loss
        else:
            loss = task_loss
        return loss, task_loss
