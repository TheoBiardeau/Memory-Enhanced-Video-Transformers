import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class GeneralADUnified(nn.Module):
    def __init__(self, config):
        """
        Unified GeneralAD model:
          - A dummy feature extractor (replace with your pretrained model).
          - A self-supervised anomaly generation module (SAG).
          - A cross-patch attention discriminator.
        
        Args:
          feature_dim (int): Dimension of patch features.
          patch_size (int): Size of each patch.
          image_size (int): Input image resolution (assumes square images).
          sag_strategy (str): 'noise_all', 'noise_random', or 'attention_shuffle'.
          noise_std (float): Standard deviation for added Gaussian noise.
          num_heads (int): Number of heads for multi-head attention.
          mlp_hidden_dim (int): Hidden dimension for the discriminator MLP.
          dropout (float): Dropout rate.
        """
        super(GeneralADUnified, self).__init__()
        # --- Feature preparation ---
        self.feature_dim = config['generalAD_config']['feature_dim']
        self.embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
        )
        self.unembedding = nn.Sequential(
            Rearrange(' b (h w)-> b h w', h = config['general_config']['feature_size'], w = config['general_config']['feature_size'])
        )
        # --- Self-Supervised Anomaly Generation (SAG) settings ---
        self.sag_strategy = config['generalAD_config']['sag_strategy']
        self.noise_std = config['generalAD_config']['noise_std']

        # --- Cross-Patch Attention Discriminator ---
        self.num_heads = config['generalAD_config']['num_heads']
        self.dropout = config['generalAD_config']['dropout']
        
        self.mha = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=self.num_heads, 
                                         dropout=self.dropout, batch_first=True)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, config['generalAD_config']['mlp_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(config['generalAD_config']['mlp_hidden_dim'], 1)  # one anomaly logit per patch
        )
        # Positional embeddings will be created dynamically based on the number of patches.
        self.pos_embedding = None

    def sag_module(self, features, attention_maps=None):
        """
        Applies self-supervised anomaly generation on patch features.
        Returns:
          features_anomaly: Distorted features.
          distortion_mask: Binary mask indicating distorted patches.
        """
        B, N, D = features.shape
        if self.sag_strategy == 'noise_all':
            noise = torch.randn_like(features) * self.noise_std
            features_anomaly = features + noise
            distortion_mask = torch.ones(B, N, device=features.device)
        elif self.sag_strategy == 'noise_random':
            mask = (torch.rand(B, N, device=features.device) > 0.5).float()
            noise = torch.randn_like(features) * self.noise_std
            features_anomaly = features + noise * mask.unsqueeze(-1)
            distortion_mask = mask
        elif self.sag_strategy == 'attention_shuffle':
            # Requires attention_maps of shape (B, num_heads, N)
            if attention_maps is None:
                raise ValueError("Attention maps must be provided for 'attention_shuffle' strategy")
            num_heads = attention_maps.shape[1]
            selected_attn = []
            for b in range(B):
                head = torch.randint(0, num_heads, (1,)).item()
                selected_attn.append(attention_maps[b, head, :])
            selected_attn = torch.stack(selected_attn, dim=0)  # (B, N)
            features_anomaly = features.clone()
            distortion_mask = torch.zeros(B, N, device=features.device)
            for b in range(B):
                k = torch.randint(1, N+1, (1,)).item()
                _, indices = torch.topk(selected_attn[b], k)
                distortion_mask[b, indices] = 1.0
                # Shuffle selected patches
                perm = torch.randperm(k)
                features_anomaly[b, indices] = features_anomaly[b, indices][perm]
        else:
            raise ValueError("Unknown SAG strategy")
        return features_anomaly, distortion_mask

    def discriminator_module(self, features):
        """
        Applies positional embedding, multi-head attention, and MLP on features.
        Returns:
          patch_scores: (B, N) anomaly scores for each patch.
        """
        B, N, D = features.shape
        if (self.pos_embedding is None) or (self.pos_embedding.shape[1] != N):
            self.pos_embedding = nn.Parameter(torch.randn(1, N, D, device=features.device))
        x = features + self.pos_embedding  # add positional information
        attn_output, _ = self.mha(x, x, x)
        x_norm = self.ln(attn_output)
        patch_scores = self.mlp(x_norm).squeeze(-1)  # (B, N)
        return patch_scores

    def forward(self, features, attention_maps=None, generate_anomaly=True):
        """
        Forward pass through the unified model.
        Args:
          x: Input images (B, 3, H, W).
          attention_maps: Optional attention maps (required for attention_shuffle).
          generate_anomaly (bool): If True, apply SAG module.
        Returns:
          patch_scores: Per-patch anomaly scores.
          distortion_mask: (Only if generate_anomaly=True) pseudo ground-truth mask.
        """
        features = self.embedding(features)
        if generate_anomaly:
            features, distortion_mask = self.sag_module(features, attention_maps)
            patch_scores = self.discriminator_module(features)
            return patch_scores, distortion_mask
        else:
            patch_scores = self.discriminator_module(features)
            patch_scores = self.unembedding(patch_scores)
            return patch_scores

