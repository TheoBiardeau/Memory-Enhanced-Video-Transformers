import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# SimpleNet complet utilisant EfficientNet-B5 comme extracteur
# -----------------------------
class SimpleNet(nn.Module):
    def __init__(self, config):
        super(SimpleNet, self).__init__()
        
        hidden_features = config['simplenet_config']['hidden_features']
        sigma = config['simplenet_config']['sigma']
        self.feature_dim = config['general_config']['feature_channel']

        # 2. Feature adaptor : une convolution 1x1 pour adapter les features par localisation
        self.feature_adaptor = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False)

        # 3. Discriminateur : MLP appliqué en 1x1 (pour traiter chaque vecteur de feature localement)
        self.discriminator = nn.Sequential(
            nn.Conv2d(self.feature_dim, hidden_features, kernel_size=1),
            nn.BatchNorm2d(hidden_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_features, 1, kernel_size=1)
        )

        # Paramètres pour le bruit et la loss tronquée
        self.sigma = sigma
        self.th_plus = 0.5
        self.th_minus = -0.5

    def forward(self, features, mode='train'):
        """
        Args:
            x (tensor): Batch d'images [B, 3, H, W]
            mode (str): 'train' renvoie les sorties pour les features normales et anormales,
                        'inference' renvoie la carte d'anomalie (négatif de la sortie du discrim.)
        """
        
        # Adaptation des features
        q = self.feature_adaptor(features)  # [B, C, H, W]

        if mode == 'train':
            # Générer des features "anormales" en ajoutant du bruit Gaussien
            noise = torch.randn_like(q) * self.sigma
            q_anom = q + noise
            s_normal = self.discriminator(q)      # [B, 1, H, W]
            s_anom = self.discriminator(q_anom)     # [B, 1, H, W]
            return s_normal, s_anom
        else:
            s = self.discriminator(q)               # [B, 1, H, W]
            # Le score d'anomalie est défini comme le négatif de la sortie du discrim.
            anomaly_map = -s
            return anomaly_map

    def compute_loss(self, s_normal, s_anom):
        """
        Calcule la loss tronquée L1 :
        loss = max(0, th_plus - s_normal) + max(0, s_anom - th_minus)
        La moyenne est prise sur tous les éléments.
        """
        loss_normal = F.relu(self.th_plus - s_normal)
        loss_anom = F.relu(s_anom - self.th_minus)
        loss = (loss_normal + loss_anom).mean()
        return loss
