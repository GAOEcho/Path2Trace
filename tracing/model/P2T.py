import torch
import torch.nn as nn


def init_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


class PathLoss(nn.Module):
    def __init__(self):
        super(PathLoss, self).__init__()

    def forward(self, input):
        sig = nn.Sigmoid()
        return sig(-torch.norm(input, 1))


class Path(nn.Module):
    def __init__(self, place_emb_dim):
        super(Path, self).__init__()
        self.s_a = nn.MultiheadAttention(embed_dim=place_emb_dim, num_heads=1)

    def forward(self, raw_path):
        att_path, weight = self.s_a(raw_path, raw_path, raw_path)
        att_path = att_path.permute(1, 0, 2).reshape([128, -1])
        return att_path, weight


class P2T(nn.Module):
    def __init__(self, place_emb_dim, path_len):
        super(P2T, self).__init__()
        self.LP = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(place_emb_dim * path_len * 2 + 2, 2000),
            # nn.Dropout(p=0.3),
            nn.Sigmoid(),
            nn.Linear(2000, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 100),
            nn.Sigmoid(),
            nn.Linear(100, 128),
            nn.Tanh(),
            # nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.LP.apply(init_normal)
        self.path1 = Path(place_emb_dim)
        self.path2 = Path(place_emb_dim)

    def forward(self, p1, p2, d1, d2):
        path1_emb, path1_weight = self.path1(p1)
        path2_emb, path2_weight = self.path2(p2)
        double_path = torch.cat((path1_emb, path2_emb, d1.unsqueeze(1), d2.unsqueeze(1)), 1)

        return self.LP(double_path), path1_emb, path2_emb, path1_weight, path2_weight
