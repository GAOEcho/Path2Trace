
import torch
import torch.nn as nn


def init_noramal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


class path_loss(nn.Module):
    def __init__(self):
        super(path_loss, self).__init__()

    def forward(self, input):
        sig = nn.Sigmoid()
        loss = sig(-torch.norm(input, 1))
        # loss = -torch.std(input)
        return loss



class path(nn.Module):
    def __init__(self):
        super(path, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(90, 200),
            nn.Sigmoid(),
            nn.Linear(200, 100),
            nn.Sigmoid(),
            nn.Linear(100, 50),
            nn.Sigmoid(),
            # nn.TransformerEncoderLayer(100, 10),
            # nn.Linear(50,45),
            # nn.Sigmoid(),
        )
        self.s_a = nn.MultiheadAttention(embed_dim=2,num_heads=1)

        # self.net.apply(init_noramal)
        # self.trans = nn.TransformerEncoderLayer()
    def forward(self, rawpath):
        # return self.s_a(raw_path,raw_path,raw_path)
        att_path,weight = self.s_a(rawpath,rawpath,rawpath)
        att_path = att_path.permute(1,0,2).reshape([128,-1])
        return att_path, weight


class loc_model(nn.Module):
    def __init__(self):
        super(loc_model, self).__init__()
        self.LP = nn.Sequential(
            nn.Linear(180+2, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            # nn.Linear(16,2), # 不用计算auc时的网络
            nn.Linear(16, 1),  # 要计算auc时的输出
            nn.Sigmoid()  # 要计算auc时的网络
        )
        self.LP.apply(init_noramal)
        self.path1 = path()
        self.path2 = path()

    def forward(self, p1, p2,d1,d2):
        path1_emb,path1_weight = self.path1(p1)
        path2_emb,path2_weight = self.path2(p2)
        double_path = torch.cat((path1_emb, path2_emb, d1.unsqueeze(1), d2.unsqueeze(1)), 1)
        return self.LP(double_path), path1_emb, path2_emb,path1_weight,path2_weight
