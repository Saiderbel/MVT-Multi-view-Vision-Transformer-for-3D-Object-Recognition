import torch
import numpy as np
from torchvision.models import alexnet, AlexNet_Weights
from torch import nn
import pytorch_lightning as pl

"""
Module containing model for per-patch operations.
References:
-----------
1. https://arxiv.org/abs/2110.13083
"""

class PerPatch(pl.LightningModule):
    def __init__(self, hidden_dimension, batch_size, num_views=12, num_patches=4*4, patch_wh=64*64) -> None:
        """
        Block for per-patch operations as introduced in [1].

        Args:
            hidden_dimension (int): Dimension of fully-connected layer output.
            batch_size (int): Size of the batch.
            num_views (int, optional): Number of views of the model. Defaults to 12.
            num_patches (int, optional): Number of patches in the image. Defaults to 4*4.
        """
        super().__init__()

        self.backbone = alexnet(weights=AlexNet_Weights.DEFAULT).features
        
        # Position embedding layer
        self.pos_embed = nn.Embedding(num_patches, hidden_dimension)

        #self.pos_indices = , device=self.device)
        self.register_buffer("pos_indices", torch.tensor([i % num_views for i in range(batch_size * num_patches * num_views)], dtype=torch.int64))
        
        # Fully connected layer, D is a hyperparameter
        self.fc = torch.nn.Linear(256, hidden_dimension)

        
    def forward(self, x):
        #input shape: B * pw * ph * 3
        batch_size = x.shape[0]

        #change to cnn compatible input: B * 3 * pw * ph
        x = x.permute(0, 3, 1, 2)
        x = self.backbone(x)
        
        #flattens unnecessary dimensions
        x = torch.reshape(x, (batch_size, 256))
        
        # Fully-connected layer
        x = self.fc(x)

        # Embed the position
        pos = self.pos_embed(self.pos_indices)
        
        return x + pos

# Test
if __name__ == '__main__':
    from src.utils.image_preprocessing.split_img import split_img
    
    img_path = '../dataset/ModelNet10/bathtub/test/bathtub_0107_snapshots/0.png'
    imgs = np.asarray(split_img(img_path))
    img_tensor = torch.tensor(np.array((imgs) / 255, dtype=np.float32))
    D = 200
    pp = PerPatch(D, batch_size=1, num_views=1)
    y = pp(img_tensor)
    print(y.shape)
        