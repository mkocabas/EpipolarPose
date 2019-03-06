from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        # nn.init.constant_(m.weight, 0.)
        # nn.init.constant_(m.bias, 0.)


class LinearPG(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5, bias=True, bn=True, leaky=False):
        super(LinearPG, self).__init__()
        self.l_size = linear_size
        self.bn = bn
        self.leaky = leaky

        if self.leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size, bias=bias)
        self.w2 = nn.Linear(self.l_size, self.l_size, bias=bias)
        self.w3 = nn.Linear(self.l_size, self.l_size, bias=bias)
        self.w4 = nn.Linear(self.l_size, self.l_size, bias=bias)

        if self.bn:
            self.batch_norm1 = nn.BatchNorm1d(self.l_size)
            self.batch_norm2 = nn.BatchNorm1d(self.l_size)
            self.batch_norm3 = nn.BatchNorm1d(self.l_size)
            self.batch_norm4 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        if self.bn:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        if self.bn:
            y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        y = self.w3(out)
        if self.bn:
            y = self.batch_norm3(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w4(y)
        if self.bn:
            y = self.batch_norm4(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = out + y

        return out


class LinearModelPG(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5,
                 input_size=15*3,
                 output_size=15*3,
                 bias=True,
                 bn=True,
                 leaky=False):
        super(LinearModelPG, self).__init__()

        self.linear_size = linear_size
        self.bn = bn
        self.leaky = leaky
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        # 2d joints
        self.input_size =  input_size
        # 3d joints
        self.output_size = output_size

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(LinearPG(self.linear_size, self.p_dropout, bias=bias, bn=self.bn,
                                               leaky=self.leaky))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w1 = nn.Linear(self.input_size, self.linear_size, bias=bias)
        self.w2 = nn.Linear(self.linear_size, self.output_size, bias=bias)
        self.w3 = nn.Linear(self.output_size, self.linear_size, bias=bias)
        self.w4 = nn.Linear(self.linear_size, self.output_size, bias=bias)

        if self.leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        if self.bn:
            self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
            self.batch_norm3 = nn.BatchNorm1d(self.linear_size)

        self.dropout = nn.Dropout(self.p_dropout)


    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        if self.bn:
            y = self.batch_norm1(y)
        y = self.relu(y)
        inp = self.dropout(y)

        s1 = self.linear_stages[0](inp)

        p1 = self.w2(s1)

        y = self.w3(p1)
        if self.bn:
            y = self.batch_norm3(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = s1 + y + inp

        y = self.linear_stages[1](y)

        y = inp + y

        p2 = self.w4(y)

        return p1, p2


def get_model(weights, **kwargs):
    model = LinearModelPG(**kwargs)
    if weights:
        model.load_state_dict(torch.load(weights)['state_dict'])
    return model


if __name__ == '__main__':
    model = LinearModelPG(input_size=16*3)

    inp = torch.randn(64,48)

    output = model(inp)

    print(output[0].shape)
