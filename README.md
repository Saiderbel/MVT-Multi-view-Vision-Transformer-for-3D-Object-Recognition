# MVT: Multi-view Vision Transformer for 3D Object Recognition

## Overview

This an implementation of the work achieved by Chen et al. in https://arxiv.org/abs/2110.13083


## How to install dependencies

To install them, run:

```
pip install -r requirements.txt
```

## How to train the model

First, specify the training configuration and all model/DataModule hyperparameters in their respective .yaml config files under configs/, then run in the root directory

```
python train.py
```

## How to test the model

After specifying the path to the checkpointed model, run in root directory:

```
python test.py
```

## Authors 
Project implemented and maintained by Mohamed Said Derbel, Alexandre Lutt, Boris Burkalo, Ludwig Gräf.