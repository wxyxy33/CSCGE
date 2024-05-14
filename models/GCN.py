import torch.nn as nn
import torch.nn.functional as F
from csgcn.models.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, outdim, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, outdim)
        self.bn = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, x, adj, non_Attributes=False):
        x = F.relu(self.bn(self.gc1(x, adj)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x



