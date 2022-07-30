import torch
from torch import nn
import pytorch_lightning as pl

"""
Module defining the structure of Transormer from Multi-view Vision Transformer introduced in [1].
References:
-----------
1.  TODO: add citation for the MVT
"""


class Transformer(pl.LightningModule):
    def __init__(self, num_input_features, num_heads, expansion_ratio, dropout=0.3):
        """
        Block of the Multi-view Vision Transformer [1].

        Args:
            num_input_features (int): Number of dimensions of the input. Denoted by D in [1]. 
            num_heads (int): Number of heads that the MSA is going to use.
            expansion_ratio (int): Expansion ratio for the MLP. Denoted by r in [1].
            dropout (float, optional): Dropout for the MLP. 
        """
        super().__init__()
        
        # First block
        # TODO: eps: https://arxiv.org/abs/2012.12877
        self.norm1 = torch.nn.LayerNorm(num_input_features, eps=1e-6)
        # TODO: batch_first? - more intuitive
        # add_bias_kv: https://arxiv.org/abs/2012.12877
        self.attention = torch.nn.MultiheadAttention(num_input_features, num_heads, batch_first=True, add_bias_kv=True)
        
        # Second block
        # TODO: is the input size correct?
        # eps: https://arxiv.org/abs/2012.12877
        self.norm2 = torch.nn.LayerNorm(num_input_features, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, int(num_input_features * expansion_ratio)),
            # https://arxiv.org/abs/2012.12877
            nn.GELU(),
            nn.Dropout(p=dropout),
            #
            torch.nn.Linear(int(num_input_features * expansion_ratio), num_input_features) 
        )
        
        # TODO: really? not sigmoid or something else?
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        # Keep the original value so we can add it to the output of MSA
        # B*V P 192
        
        # TODO: Usage of torch.clone?
        x_orig = torch.clone(x)
        
        # Start of the first block - normalize
        x = self.norm1(x)
        # Multihead Self-attention (TODO: We probably don't need the weights)
        x, _ = self.attention(x, x, x)
        # Add the original values to the output of MSA
        x = x_orig + x

        # Keep the output of MSA for later to add it to the output of MLP
        # TODO: Usage of torch.clone?
        x_msa = torch.clone(x)
        
        # Start of the second layer - normalize
        x = self.norm2(x)
        # Run data through the Multi-level Perceptron
        x = self.mlp(x)
        
        # Add the output of MSA to the output of MLP
        x = x + x_msa
        
        return x

# Test
if __name__ == '__main__':
    num_features = 200
    num_heads = 8
    expansion_ratio = 3
    transformer = Transformer(num_features, num_heads, expansion_ratio)
    x = torch.rand(1, 1, num_features)
    y = transformer(x)