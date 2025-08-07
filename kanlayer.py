import torch
import torch.nn as nn
import torch.nn.functional as F

from allkan import KANInterfaceV2


class MoKLayer(nn.Module):
    def __init__(self, in_features, out_features, expert_config, res_con=False, with_bn=False, with_dropout=False):
        super(MoKLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_expert = len(expert_config)

        self.gate = nn.Linear(in_features, self.n_expert)
        self.softmax = nn.Softmax(dim=-1)
        self.experts = nn.ModuleList(
            [KANInterfaceV2(in_features, out_features, k[0], k[1]) for k in expert_config])

        self.res_con = res_con
        self.with_bn = with_bn
        self.with_dropout = with_dropout
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        score = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.n_expert)], dim=-1)  # (BxN, Lo, E)
        # add BN here? Naming Expert Norm?
        y = torch.einsum("BLE,BE->BL", expert_outputs, score)

        if self.res_con:
            y = x + y
        if self.with_bn:
            y = self.bn(y)
        if self.with_dropout:
            y = self.dropout(y)
        return y