import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from models.features_extractors import EfficientNet_feature_B5
from tqdm import tqdm

# ---------------------------
# PatchCore class as defined earlier.
# ---------------------------
class PatchCore:
    def __init__(self, config, device='cuda'):
        """
        Initialize the PatchCore algorithm.
        
        Args:
            backbone: A pretrained feature extractor that outputs feature maps.
            patch_size: The size of the local patch (neighborhood) used for feature aggregation.
            stride: Stride for sliding over the feature map.
            coreset_target: Target number of patch features after coreset subsampling.
            proj_dim: Target dimension for random projection.
            device: Device to run computations ('cuda' or 'cpu').
        """
        self.backbone = EfficientNet_feature_B5(config).to(device)
        self.patch_size = config['patch_core']['patch_size']
        self.stride = config['patch_core']['stride']
        self.coreset_target = config['patch_core']['coreset_target']
        self.device = 'cuda'
        self.memory_bank = None
        self.proj_dim = config['patch_core']['proj_dim']
        # Create a random projection matrix with shape (feature_dim, proj_dim)
        self.random_proj_matrix = torch.randn((config['general_config']['feature_channel'], self.proj_dim)).to(device)
        


    def extract_patch_features(self, feature_map):
        """
        Extract locally aware patch features from an image and return the grid shape.
        
        Args:
            image: A torch tensor of shape (C, H, W).
            
        Returns:
            patches: A tensor of shape (num_patches, C) containing aggregated patch features.
            grid_shape: A tuple (grid_H, grid_W) indicating the number of patches along height and width.
        """
        # Compute feature map using the pretrained backbone.
        feature_map = feature_map.squeeze(0)  # (C, H', W')
        C, H, W = feature_map.shape
        
        # Compute grid dimensions based on patch_size and stride.
        grid_H = (H - self.patch_size) // self.stride + 1
        grid_W = (W - self.patch_size) // self.stride + 1
        
        patches = []
        # Slide a window over the feature map.
        for h in range(0, H - self.patch_size + 1, self.stride):
            for w in range(0, W - self.patch_size + 1, self.stride):
                patch = feature_map[:, h:h+self.patch_size, w:w+self.patch_size]
                # Aggregate patch features using adaptive average pooling.
                patch_vec = F.adaptive_avg_pool2d(patch, (1, 1)).view(-1)
                patches.append(patch_vec)
                
        patches = torch.stack(patches)  # Shape: (num_patches, C)
        return patches, (grid_H, grid_W)

    def build_memory_bank(self, train_images):
        """
        Build the patch-feature memory bank from nominal training images.
        
        Args:
            train_images: List of training images (each as a torch tensor).
        """
        t = tqdm(total=len(train_images), desc="Memory Training")
        memory_features = []
        for i ,img in enumerate( train_images):
            t.update()
            patches, _ = self.extract_patch_features(img)
            memory_features.append(patches)
            if i == 1000 : break
        self.memory_bank = torch.cat(memory_features, dim=0)  # Shape: (N, C)

    def random_projection(self, features):
        """
        Apply random projection to reduce feature dimensionality.
        
        Args:
            features: Tensor of shape (N, C).
            
        Returns:
            Projected features of shape (N, proj_dim).
        """
        return features @ self.random_proj_matrix

    def greedy_coreset_selection(self):
        """
        Réduit la banque de mémoire en utilisant une sélection de coreset gloutonne vectorisée.
        Cette version met à jour pour chaque échantillon la distance minimale vers le coreset actuel,
        et sélectionne ensuite le point qui maximise cette distance.
        """
        if self.memory_bank is None:
            raise ValueError("La banque de mémoire n'est pas construite.")

        M = self.memory_bank
        N = M.shape[0]
        # Projection aléatoire pour réduire la dimensionnalité.
        M_proj = self.random_projection(M)  # Shape: (N, proj_dim)
        
        # Initialiser avec le premier indice arbitrairement.
        selected_indices = [0]
        
        # Calculer la distance de tous les points au premier point sélectionné.
        diff = M_proj - M_proj[0].unsqueeze(0)  # Shape: (N, proj_dim)
        min_dists = torch.norm(diff, dim=1)  # Vecteur de forme (N,)

        # Itérer jusqu'à atteindre le nombre cible.
        for i in range(1, self.coreset_target):
            # Trouver l'indice du candidat qui a la distance minimale maximale.
            best_idx = torch.argmax(min_dists).item()
            # Si la distance maximale est nulle, on arrête.
            if min_dists[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            # Mettre à jour les distances minimales pour chaque point en comparant avec le nouveau point sélectionné.
            new_dists = torch.norm(M_proj - M_proj[best_idx].unsqueeze(0), dim=1)
            min_dists = torch.minimum(min_dists, new_dists)
        
        # Mettre à jour la banque de mémoire avec le coreset sélectionné.
        self.memory_bank = M[selected_indices]

    def compute_anomaly_score(self, test_image):
        """
        Compute the anomaly score for a test image.
        
        Args:
            test_image: A torch tensor representing the test image.
            
        Returns:
            s_star: The image-level anomaly score (scalar).
            patch_scores: A tensor of patch-level anomaly scores (1D).
            grid_shape: The grid shape (grid_H, grid_W) for reshaping patch_scores.
        """
        test_patches, grid_shape = self.extract_patch_features(test_image)
        if self.memory_bank is None:
            raise ValueError("Memory bank is not built.")
        
        M = self.memory_bank.to(test_patches.device)
        # Compute pairwise Euclidean distances between test patches and memory bank features.
        dists = torch.cdist(test_patches, M, p=2)  # Shape: (N_test, N_memory)
        min_dists, _ = torch.min(dists, dim=1)  # Distance for each patch.
        s_star, _ = torch.max(min_dists, dim=0)
        return s_star.item(), min_dists, grid_shape

    def compute_anomaly_map(self, test_image):
        """
        Compute and reshape the anomaly scores into a 2D anomaly map.
        
        Args:
            test_image: A torch tensor representing the test image.
            
        Returns:
            s_star: The image-level anomaly score (scalar).
            anomaly_map: A 2D tensor (grid_H x grid_W) suitable for visualization with matplotlib.
        """
        s_star, patch_scores, grid_shape = self.compute_anomaly_score(test_image)
        anomaly_map = patch_scores.view(grid_shape)
        return s_star, anomaly_map

