import torch.nn as nn
import torch
from csgcn.models.GCN import GCN
import torch.nn.functional as F


class GCN_C(nn.Module):
    def __init__(self, nfeat, nhid, outdim, dropout, args):
        super(GCN_C, self).__init__()
        self.gcn = GCN(nfeat, nhid, outdim, dropout)
        print('load pretrain weight from [{}]'.format(args.work_path+'/model_weights/'+ args.dataset +'/triplet-pretrain-GCN(k='+ str(args.k_num) +')('+ str(args.round_num) +").pth"))
        self.gcn.load_state_dict(torch.load(args.work_path+'/model_weights/'+ args.dataset +'/triplet-pretrain-GCN(k='+ str(args.k_num) +')('+ str(args.round_num) +").pth"))

        self.v = 1.0
        self.cluster_layer = nn.Parameter(torch.Tensor(args.clusters_num, nfeat))

    def forward(self, x, adj):
        x = self.gcn(x, adj)
        
        # x = F.normalize(x)
        # Q = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # Q = Q.pow((self.v + 1.0) / 2.0)
        # Q = (Q.t() / torch.sum(Q, 1)).t()
        
        # S = torch.stack([1 - F.cosine_similarity(torch.unsqueeze(i, 0), self.cluster_layer) for i in x], 0)
        # Q = 1.0 / (1.0 + S / self.v)
        # Q = Q.pow((self.v + 1.0) / 2.0)
        # Q = (Q.t() / torch.sum(Q, 1)).t()
        
        x = F.normalize(x)
        # S = 1 - torch.mm(x, self.cluster_layer.t())
        if x.dtype != self.cluster_layer.dtype:
            x = x.to(self.cluster_layer.dtype)

        S = 1 - torch.mm(x, self.cluster_layer.t())


        
        Q = 1.0 / (1.0 + S / self.v)
        Q = Q.pow((self.v + 1.0) / 2.0)
        Q = (Q.t() / torch.sum(Q, 1)).t()
        
        return Q, x
        # return x















