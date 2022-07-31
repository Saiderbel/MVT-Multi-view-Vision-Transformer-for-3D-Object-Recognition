import torch
from torch import nn
from src.models.components.per_patch import PerPatch
from src.models.components.transformer import Transformer
from src.models.components.classifier import Classifier
from typing import Any, List
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MaxMetric
import torch
from pytorch_lightning import LightningModule

"""
Module defining the structure of Multi-view Vision Transformer introduced in [1].
References:
-----------
[1] https://arxiv.org/abs/2110.13083
"""
class MVT(LightningModule):

    def __init__(
        self,
        lr,
        batch_size,
        expansion_ratio,
        weight_decay=0.05,
        transformer_dropout=0.3,
        classifier_dropout=0.2,
        hidden_dimension=192,
        patch_wh=64 * 64,
        num_patches=4 * 4,
        attention_heads=3,
        local_blocks=8,
        global_blocks=4,
        num_views=12,
        nbr_classes=10
    ):
        """
        Full model introduced in Multi-view Vision Transformer introduced in [1]. All the default parameters
        are taken from the "tiny" MVT experiment from [1].

        Args:
            expansion_ratio (int): Expansion ratio for transformer block.
            hidden_dimension (int, optional): Hidden dimension of for the output dimension of the patch block and input
            dimension of attention blocks. Defaults to 192.
            patch_embed_size (int, optional): Size of patch embedding. Defaults to 64*64.
            patch_wh (int, optional): Width * height of patch. Defaults to 64*64.
            num_patches (int, optional): Number of patches per view. Defaults to 4*4.
            attention_heads (int, optional): Number of heads for local and global transformer. Defaults to 3.
            local_blocks (int, optional): Number of local attention blocks. Defaults to 8.
            global_blocks (int, optional): Number of global attention blocks. Defaults to 4.
            nbr_views (int, optional): Number of views of the 3D model. Defaults to 12.
            nbr_classes (int, optional): Number of possible classes of the models. Defaults to 10.
        """

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Per patch operations block - TODO: for each view???
        self.per_patch = PerPatch(hidden_dimension, batch_size=batch_size, num_views=num_views,
                                  num_patches=num_patches, patch_wh=patch_wh)

        # TODO: is the embedding correct?
        self.view_embed = nn.Embedding(num_views, hidden_dimension)
        self.register_buffer("view_indices",
                             torch.tensor([i for i in range(num_views)], dtype=torch.int64))

        # Local transformers
        self.local_transformers = torch.nn.ModuleList()
        for _ in range(local_blocks):
            self.local_transformers.append(Transformer(hidden_dimension, attention_heads, expansion_ratio, dropout=transformer_dropout))

        # Global transformers
        self.global_transformers = torch.nn.ModuleList()
        for _ in range(global_blocks):
            self.local_transformers.append(Transformer(hidden_dimension, attention_heads, expansion_ratio, dropout=transformer_dropout))

        # Classifier block
        self.classifier = Classifier(batch_size=batch_size, num_views=num_views, num_patches=num_patches, 
                                     nbr_classes=nbr_classes, hidden_dimension=hidden_dimension, dropout=classifier_dropout)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_acc_best = MaxMetric()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor):
        """
        Forward function for the MVT [1].

        Args:
            x (ndarray): Tensor of shape (B, num_views, num_patches, patch_width, patch_height, 3)
        """
        # (B, num_views, num_patches, patch_width, patch_height, 3)
        # print('x size:', x.size())

        # Create empty array for Per Patch output
        x = torch.empty((self.hparams.batch_size * self.hparams.num_views * self.hparams.num_patches, self.hparams.hidden_dimension))

        # Flatten the array so it can be batched and there's no need for
        # iterating over samples.

        input = torch.flatten(input, end_dim=2)

        # (B * num_views * num_patches, pw, ph, 3)
        # print('Flat x size:', x.size())

        # Run x through the Per Patch block
        x = self.per_patch(input)

        # Unflatten x by one dimension (TODO: Hopefully correct order?)
        x = torch.reshape(x, (self.hparams.batch_size, self.hparams.num_views, self.hparams.num_patches, self.hparams.hidden_dimension))

        # Get view vectors and reshape them so they can be concatenated
        views = self.view_embed(self.view_indices).unsqueeze(dim=1).unsqueeze(dim=0)
        views = torch.cat([views] * self.hparams.batch_size)

        # Concatenate the view embeddings and values
        x = torch.cat((views, x), dim=2)

        # Flatten back to the size: (B*V, P, hidden_dimension)
        x = torch.flatten(x, end_dim=1)

        # Send it to local transformers
        for local_transformer in self.local_transformers:
            x = local_transformer(x)

        # (B * num_views, num_patches + 1, hidden_dimension)
        # print('Local transformer shape:', x.size())

        # Reshape back into: (B, num_views, num_patches + 1, hidden_dimension)
        x = torch.reshape(x, (self.hparams.batch_size, self.hparams.num_views, self.hparams.num_patches + 1, self.hparams.hidden_dimension))

        # Flatten into shape: (B, num_views * (num_patches + 1), hidden_dimension)
        x = torch.flatten(x, start_dim=1, end_dim=2)

        # Run it through the global transformers
        for global_transformer in self.global_transformers:
            x = global_transformer(x)

        # (B, num_views * (num_patches + 1), hidden_dimension)
        # print('Global transformer shape:', x.size())
        
        # Classify the images
        x = self.classifier(x)

        return x

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch["view_patches"], batch["labels"]
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optim =  torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", patience=5, factor=0.5)
        return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": "val/acc"}
    
    # Only testing for fuckups    
    def dimension_manip_check(self, x):
        # manipulation check
        print('manip1:')
        print(x.size())
        x_orig = torch.clone(x)
        x = torch.flatten(x, end_dim=2)
        print(x.size())
        x = torch.flatten(x, start_dim=1)
        print(x.size())
        x = torch.reshape(x, (2, 12, 16, 64, 64, 3))
        print(x.size())
        print((x_orig == x).all())
        
        # manipulation check 2
        print('manip2:')
        x_orig = torch.flatten(x_orig, end_dim=2)
        x_orig = self.per_patch(x_orig)
        x = torch.clone(x_orig)
        print(x.size())
        x = torch.reshape(x, (self.hparams.batch_size, self.hparams.num_views, self.hparams.num_patches, self.hparams.hidden_dimension))
        print(x.size())
        x = torch.flatten(x, end_dim=1)
        print(x.size())
        x = torch.reshape(x, (self.hparams.batch_size, self.hparams.num_views, self.hparams.num_patches, self.hparams.hidden_dimension))
        print(x.size())
        x = torch.flatten(x, start_dim=1, end_dim=2)
        print(x.size())
        x = torch.reshape(x, (self.hparams.batch_size * self.hparams.num_views * self.hparams.num_patches, self.hparams.hidden_dimension))
        print(x.size())
        print(x[0][0])
        print(x_orig[0][0])
        print((x_orig == x).all())


# Test
if __name__ == '__main__':
    import numpy as np
    from src.datamodules.modelnet_datamodule import ModelNetSnapshot
    
    ds = ModelNetSnapshot('test', dataset_path='data/ModelNet10')
    
    sample1 = ds[0]['view_patches'].unsqueeze(dim=0)
    sample2 = ds[1]['view_patches'].unsqueeze(dim=0)
    x = torch.cat((sample1, sample2))
    print(x.size())
    
    mvt = MVT(batch_size=2, expansion_ratio=2)
    y = mvt(x)
    print(y)
    mvt.dimension_manip_check(x)
    
    