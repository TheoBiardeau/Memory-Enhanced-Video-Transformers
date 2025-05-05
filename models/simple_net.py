import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, config):
        super(SimpleNet, self).__init__()
        
        hidden_features = config['simplenet_config']['hidden_features']
        sigma = config['simplenet_config']['sigma']
        self.feature_dim = config['general_config']['feature_channel']

        self.feature_adaptor = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False)

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.feature_dim, hidden_features, kernel_size=1),
            nn.BatchNorm2d(hidden_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_features, 1, kernel_size=1)
        )

        self.sigma = sigma
        self.th_plus = 0.5
        self.th_minus = -0.5

    def forward(self, features, mode='train'):

        q = self.feature_adaptor(features)  # [B, C, H, W]

        if mode == 'train':
            noise = torch.randn_like(q) * self.sigma
            q_anom = q + noise
            s_normal = self.discriminator(q)      # [B, 1, H, W]
            s_anom = self.discriminator(q_anom)     # [B, 1, H, W]
            return s_normal, s_anom
        else:
            s = self.discriminator(q)               # [B, 1, H, W]
            anomaly_map = -s
            return anomaly_map

    def compute_loss(self, s_normal, s_anom):
        loss_normal = F.relu(self.th_plus - s_normal)
        loss_anom = F.relu(s_anom - self.th_minus)
        loss = (loss_normal + loss_anom).mean()
        return loss
