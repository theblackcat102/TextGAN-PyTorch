# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : RelGAN_D.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm

from models.discriminator import CNNDiscriminator

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [300, 300, 300, 300]

def cacl_gradient_penalty(net_D, real, fake):
    t = torch.rand(real.size(0), 1, 1).to(real.device)
    t = t.expand(real.size())

    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)
    return loss_gp



class GradNorm(nn.Module):
    def __init__(self, *modules):
        super(GradNorm, self).__init__()
        self.main = nn.Sequential(*modules)

    def forward(self, x):
        x.requires_grad_(True)
        fx = self.main(x)
        fx = fx.reshape(x.shape[0], -1)

        grad_x = torch.autograd.grad(
            fx, x, torch.ones_like(fx), create_graph=True,
            retain_graph=True)[0]


        grad_norm = torch.norm(grad_x.view(grad_x.size(0), -1), dim=1)
        grad_norm = grad_norm.view(-1, *[1 for _ in range(len(fx.shape) - 1)])


        fx = (fx / (grad_norm + torch.abs(fx)))
        return fx.squeeze(1)

class RelGAN_D(CNNDiscriminator):
    def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size, padding_idx, norm='none',gpu=False, dropout=0.25):
        super(RelGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx,
                                       gpu, dropout)

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)
        self.num_rep = num_rep

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)
        if norm == 'spectral':
            print('use spectral')
            self.convs = nn.ModuleList([
                spectral_norm(nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single))) for (n, f) in
                zip(dis_num_filters, dis_filter_sizes)
            ])        
        else:
            self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)),
                )
                     for (n, f) in
                zip(dis_num_filters, dis_filter_sizes)
            ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits

    