import torch.nn as nn
import torch
import pytorch_lightning as pl

"""
Module defining the structure of Multi-view Vision Transformer introduced in [1].
References:
-----------
1.  TODO: add citation for the MVT
"""

class Classifier(pl.LightningModule):

    def __init__(self, batch_size,
                 num_views=12,
                 num_patches=16,
                 nbr_classes=10,
                 hidden_dimension=192,
                 dropout=0.2):
        """
        Classifier of the MVT architecture as per [1].

        Args:
            batch_size (int): Batch size of the input.
            num_views (int, optional): Number of views of the 3D model. Defaults to 12.
            num_patches (int, optional): Number of patches in each view. Defaults to 16.
            nbr_classes (int, optional): Number of classes of the dataset. Defaults to 10.
            hidden_dimension (int, optional): Hidden dimension. Defaults to 192.
        """
        super().__init__()

        
        self.num_views = num_views
        self.num_patches = num_patches
        self.hidden_dimension = hidden_dimension
        self.batch_size = batch_size
        

        self.avg_pooling = torch.nn.AvgPool1d(kernel_size=num_views)
        self.classifier = nn.Sequential(
            torch.nn.Linear(in_features=hidden_dimension, out_features=nbr_classes),
            torch.nn.Dropout(p=dropout)
        )

    def forward(self, x):
        """
        Forward function for the classifier.

        Args:
            x (ndarray): Numpy array in shape (B, num_views * (num_patches + 1), hidden_dimesion)

        Returns:
            ndarray: Probabilities of the classes.
        """
        # TODO: Check functionality!!
        # input should be of shape (C, num_views) = (num_views*(w*h+1), hidden_dimension)
        # sum pooling only over the attended special tokens

        # Take every 17th element of the tensor, after x = {m_0^j}_{j=1, num_views}
        x = x[:, 0:self.num_views * self.num_patches:self.num_patches, :]
        x = x.reshape(self.hidden_dimension * self.batch_size, self.num_views)
                
        # Do the sum pooling
        x = self.avg_pooling(x)
                
        # Classify the input
        x = x.reshape(self.batch_size, self.hidden_dimension)
        x = self.classifier(x)
        
        return x


## Test
if __name__ == '__main__':
    num_views = 12 * 1
    num_patches = 16
    nbr_class = 10
    hidden_dimension = 192
    batch_size = 2
    
    model = Classifier(batch_size=batch_size)
    x = torch.rand(batch_size, num_views * (num_patches + 1), hidden_dimension)
    y = model(x)
    print(y)
