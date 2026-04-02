# Modified from https://github.com/AnTao97/dgcnn.pytorch/blob/master/main_semseg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature      # (batch_size, 2*num_dims, num_points, k)

class DGCNNFeat(nn.Module):
    def __init__(self, 
                 emb_dims=512,
                 k=20,
                 dropout=0.5,
                 global_feat=True):
        super(DGCNNFeat, self).__init__()
        self.k = k  # number of neighbors, default is 20
        self.emb_dims = emb_dims
        self.global_feat = global_feat

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(448, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.emb_dims*2, self.emb_dims, bias=False)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

    def forward(self, x):  # [b, 1024, 3]
        x = x.transpose(1, 2) # [b, 3, 1024]
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        ps_feat = self.conv4(x)  # point wise feature
        x1 = F.adaptive_max_pool1d(ps_feat, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(ps_feat, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn5(self.linear1(x)), negative_slope=0.2)
        if self.global_feat:
            return x
        return ps_feat.transpose(1, 2), x


class DGCNNVAEEncoder(nn.Module):
    def __init__(self, k=20, local_dim=256, zdim=256):
        super().__init__()
        self.local_dim = local_dim
        self.zdim = zdim
        # DGCNN backbone 输出 (B, N, local_dim)
        self.backbone = DGCNNFeat(k=k, emb_dims=local_dim, global_feat=False)

        self.fc_m = nn.Linear(local_dim, zdim)
        self.bn_m  = nn.BatchNorm1d(zdim)
        self.fc_v = nn.Linear(local_dim, zdim)
        self.bn_v  = nn.BatchNorm1d(zdim)

    def forward(self, x):
        local_feat, global_feat = self.backbone(x)                  # (B, local_dim)
        z_mu = F.relu(self.bn_m(self.fc_m(global_feat)))  # (B, zdim)
        z_logvar = F.relu(self.bn_v(self.fc_v(global_feat)))  # (B, zdim)

        return local_feat, z_mu, z_logvar

    
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    args.cuda = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda:0'
    model = DGCNNFeat(emb_dims=256).to(device)
    model = model.eval()
    x = torch.randn(8, 2048, 3).cuda()
    ps_feat = model(x)
    print(ps_feat.shape)