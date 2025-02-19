# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

from .utils import get_mask, shuffle, spectral_norm
from .utils import MLP, Disc, TextCNN

class TextEncoder(nn.Module):
    """
        TextCNN Encoder
        :param vocab_size: int, size of vocabulary
        :param hidden_dim: int, size of word and question embedding
        :param dropout: float, dropout rate
        :param wv: Tensor of (vocab_size, hidden_dim), initial word embedding, default = None
    """

    def __init__(self, vocab_size, hidden_dim, dropout, wv=None):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        if wv is not None:
            self.embed.weight.data.copy_(wv)
        self.encoder = TextCNN(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        return
    
    def forward(self, input):
        """
            :param input: Tensor of long (batch_size, seq_len), word index
            :returns: (embed, hidden)
                embed: Tensor of (batch_size, seq_len, hidden_dim), word embedding
                hidden: Tensor of (batch_size, hidden_dim), question embedding
        """
        embed = self.embed(input)
        embed_dp = self.dropout(embed)
        hidden = self.encoder(embed_dp)
        return embed, hidden

class AttnModel(nn.Module):
    """
        Attention Model
        :param dim: int, size of hidden vector
        :param dropout: float, dropout rate of attention model
    """

    def __init__(self, dim, dropout):
        super(AttnModel, self).__init__()
        self.score = MLP(dim*2, 1, dim, 0, n_layers=2, act=torch.tanh)
        self.fw = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=dropout)
        return
    
    def forward(self, q, k, v=None, mask=None):
        """
            :param q: Tensor of (batch_size, hidden_size) or (batch_size, out_size, hidden_size), query, out_size = 1 if discarded
            :param k: Tensor of (batch_size, in_size, hidden_size), key
            :param v: Tensor of (batch_size, in_size, hidden_size), value, default = None, means v = k
            :param mask: Tensor of (batch_size, in_size), key/value mask, where 0 means data and 1 means pad, default = None, means zero matrix
            :returns: (output, attn)
                output: Tensor of (batch_size, hidden_size) or (batch_size, out_size, hidden_size), attention output, shape according to q
                attn: Tensor of (batch_size, in_size) or (batch_size, out_size, in_size), attention weight, shape according to q
        """
        if v is None:
            v = k
        # (b, h) -> (b, o, h)
        q_dim = q.dim()
        if q_dim == 2:
            q = q.unsqueeze(1)

        # (b, o, i, h)
        output_size = q.size(1)
        input_size = k.size(1)
        qe = q.unsqueeze(2).expand(-1, -1, input_size, -1)
        ke = k.unsqueeze(1).expand(-1, output_size, -1, -1)
        # (b, o, i, h) -> (b, o, i, 1) -> (b, o, i)
        score = self.score(torch.cat((qe, ke), dim=-1)).squeeze(-1)
        if mask is not None:
            score.masked_fill_(mask.unsqueeze(1).expand(-1, output_size, -1), -float('inf'))
        attn = F.softmax(score, dim=-1)
        # (b, o, i) * (b, i, h) -> (b, o, h)
        output = torch.bmm(attn, v)
        
        # (b, o, h) -> (b, h), (b, o, i) -> (b, i)
        if q_dim == 2:
            output = output.squeeze(1)
            attn = attn.squeeze(1)
        
        output = F.leaky_relu(self.fw(self.dropout(output)))
        return output, attn

class ConceptEstimator(nn.Module):
    """
        Concept estimator
        min Classifier_loss

        :param hidden_dim: int, size of question embedding
        :param concept_size: int, number of concept classes
        :param pos_weight: float, positive sample weight in unbalanced multi-label concept classifier
        :param dropout: float, dropout rate
    """

    def __init__(self, hidden_dim, concept_size, pos_weight, dropout):
        super(ConceptEstimator, self).__init__()
        self.classifier = MLP(hidden_dim, concept_size, hidden_dim, dropout, n_layers=2)
        pos_weight = torch.ones(concept_size) * pos_weight
        self.loss = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        return
    
    def forward(self, hidden, label):
        """
            :param hidden: Tensor of (batch_size, hidden_dim), question embedding
            :param label: Tensor of long (batch_size, concept_size), question concept one-hot label
            :returns: Tensor of (), BCE loss of concept label
        """
        output = self.classifier(hidden)
        loss = self.loss(output, label.float())
        return loss

class MIEstimator(nn.Module):
    """
        MI estimator between question and word embedding for maximization
        max MI(X,Y)

        :param embed_dim: int, size of word embedding
        :param hidden_dim: int, size of question embedding
        :param dropout: float, dropout rate
    """

    def __init__(self, embed_dim, hidden_dim, dropout):
        super(MIEstimator, self).__init__()
        self.disc = Disc(embed_dim, hidden_dim, dropout)
        return
    
    def shuffle_in_batch(self, x):
        # |0 0 0 0 0|    |1 2 3 1 2|
        # |1 1 1 1 1| => |2 3 0 2 3|
        # |2 2 2 2 2|    |3 0 1 3 0|
        # |3 3 3 3 3|    |0 1 2 0 1|
        device = x.device
        batch_size, seq_len, _ = x.size()
        if batch_size == 1:
            sx = x
        else:
            batch_index = torch.arange(batch_size).unsqueeze(1)
            shuffle_index = (torch.arange(batch_size - 1) + 1).repeat(seq_len // (batch_size - 1) + 1)[:seq_len].unsqueeze(0)
            batch_index = batch_index.to(device)
            shuffle_index = shuffle_index.to(device)
            shuffled_batch_index  = (batch_index + shuffle_index) % batch_size
            seq_batch_index = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            seq_batch_index = seq_batch_index.to(device)
            sx = x[shuffled_batch_index, seq_batch_index]
        return sx
    
    def forward(self, embed, hidden, lengths):
        """
            average MI between hidden and each word embedding
            MI(X,Y) = Avg(E_pxy[-sp(-T(xi,y))] - E_pxpy[sp(T(xi,y))])

            :param embed: Tensor of (batch_size, seq_len, embed_dim), word embedding vector
            :param hidden: Tensor of (batch_size, hidden_dim), question hidden vector
            :param lengths: Tensor of (batch_size), length of valid words in each question
            :returns: Tensor of (), mi between question and each word embedding
        """
        # batch_size * seq_len * hidden_dim
        seq_len = embed.size(1)
        hidden = hidden.unsqueeze(1).expand(-1, seq_len, -1)
        s_hidden = self.shuffle_in_batch(hidden)
        # batch_size * seq_len, [F F F T T T], False means data, True means pad
        mask = get_mask(seq_len, lengths)
        # seq_len
        seq_batch_len = (~mask).sum(dim=0)
        # P(X,Y) = (x, y), P(X)P(Y) = (x, sy)
        # batch_size * seq_len -> seq_len
        # -sp(disc(x, y)).mean() - sp(disc(x, sy)).mean()
        seq_mi = -F.softplus(-self.disc(embed, hidden)).masked_fill(mask, 0).sum(dim=0) / seq_batch_len - F.softplus(self.disc(embed, s_hidden)).masked_fill(mask, 0).sum(dim=0) / seq_batch_len
        # seq_len -> 1
        mi = seq_mi.mean()
        return mi

class DisenEstimator(nn.Module):
    """
        Disentangling estimator by WGAN-like adversarial training and spectral normalization for MI minimization
        MI(X,Y) = E_pxy[T(x,y)] - E_pxpy[T(x,y)]
        min_xy max_T MI(X,Y)

        :param hidden_dim: int, size of question embedding
        :param dropout: float, dropout rate
    """

    def __init__(self, hidden_dim, dropout):
        super(DisenEstimator, self).__init__()
        self.disc = Disc(hidden_dim, hidden_dim, dropout)
        return
    
    def forward(self, x, y):
        """
            :param x: Tensor of (batch_size, hidden_dim), x
            :param y: Tensor of (batch_size, hidden_dim), y
            :returns: Tensor of (), loss for MI minimization
        """
        sy = shuffle(y)
        loss = self.disc(x, y).mean() - self.disc(x, sy).mean()
        return loss
    
    def spectral_norm(self):
        """
            spectral normalization to satisfy Lipschitz constrain for Disc of WGAN
        """
        # Lipschitz constrain for Disc of WGAN
        with torch.no_grad():
            for w in self.parameters():
                w.data /= spectral_norm(w.data)
        return

class VectorMILoss(nn.Module):
    """
        MI estimator between hidden vectors for maximization
        max MI(X,Y)

        :param hidden_dim: int, size of hidden vector
        :param dropout: float, dropout rate
    """

    def __init__(self, hidden_dim, dropout):
        super(VectorMILoss, self).__init__()
        self.disc = Disc(hidden_dim, hidden_dim, dropout)
        return
    
    def forward(self, x, y):
        """
            max MI(X,Y) = E_pxy[-sp(-T(x,y))] - E_pxpy[sp(T(x,y))]

            :param x: Tensor of (batch_size, hidden_dim), x
            :param y: Tensor of (batch_size, hidden_dim), y
            :returns: Tensor of (), loss for MI maximization
        """
        sy = shuffle(y)
        mi = -F.softplus(-self.disc(x, y)).mean() - F.softplus(self.disc(x, sy)).mean()
        loss = -mi
        return loss
