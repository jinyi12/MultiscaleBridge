<h1 align='center'>Stochastic Optimal Control for Diffusion Bridges in Function Spaces (DBFS)</h1>
<div align="center">
  <a href="https://bw-park.github.io/" target="_blank">Byoungwoo Park</a><sup>1</sup>&ensp;<b>&middot;</b>&ensp;
  <a href="https://jungwon-choi.github.io/" target="_blank">Jungwon Choi</a><sup>1</sup>&ensp;<b>&middot;</b>&ensp;
  <a href="https://www.sungbin-lim.net/" target="_blank">Sungbin Lim</a><sup>2,3</sup>&ensp;<b>&middot;</b>&ensp;
  <a href="https://juho-lee.github.io/" target="_blank">Juho Lee</a><sup>1</sup><br>
  <sup>1</sup>KAIST &emsp; <sup>2</sup>Korea University &emsp; <sup>3</sup>LG AI Research<br>
</div>
<br>
<p align="center">
  <a href="https://arxiv.org/abs/2405.20630">
    <img src="https://img.shields.io/badge/neurips-blue" alt="neurips">
  </a>
</p>

[Stochastic Optimal Control for Diffusion Bridges in Function Spaces](https://arxiv.org/abs/2405.20630) (**DBFS**) extends previous bridge matching algorithms to learn diffusion models between two infinite-dimensional distributions in a resolution-free manner.

## Examples
| pi_0 ⇆ pi_T | Results (left: pi_0 → pi_T, right: pi_0 ← pi_T) |
|-------------------------|-------------------------|
| EMNIST ⇆ MNIST (32x32, observed) | <p float="left"> <img src="./assets/emnist2mnist_32.gif" alt="drawing" width="180"/>  <img src="./assets/mnist2emnist_32.gif" alt="drawing" width="180"/> </p>  |
| EMNIST ⇆ MNIST (64x64, unseen) | <p float="left"> <img src="./assets/emnist2mnist_64.gif" alt="drawing" width="180"/>  <img src="./assets/mnist2emnist_64.gif" alt="drawing" width="180"/> </p>  |
| EMNIST ⇆ MNIST (128x128, unseen) | <p float="left"> <img src="./assets/emnist2mnist_128.gif" alt="drawing" width="180"/>  <img src="./assets/mnist2emnist_128.gif" alt="drawing" width="180"/> </p>  |
| AFHQ-64 Wild ⇆ Cat (64x64, observed) | <p float="left"> <img src="./assets/wild2cat_64.gif" alt="drawing" width="180"/>  <img src="./assets/cat2wild_64.gif" alt="drawing" width="180"/> </p> | 
| AFHQ-64 Wild ⇆ Cat (128x128, unseen) | <p float="left"> <img src="./assets/wild2cat_128.gif" alt="drawing" width="180"/>  <img src="./assets/cat2wild_128.gif" alt="drawing" width="180"/> </p> | 

## Installation
This code is developed with Python3 and Pytorch. To set up an environment with the required packages,

1. Create a virtual enviornment, for example:
```
conda create -n dbfs pip
conda activate dbfs
```
2. Install Pytorch according to the [official instructions](https://pytorch.org/get-started/locally/).
3. Install the requirements:
```
pip install -r requirements.txt
```

## Download AFHQ dataset
<!-- https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq -->
Download the AFHQ dataset from [stargan-v2](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq), and save them in the `dbfs/data` directory.

You can also download the dataset with the following commands:
```
bash download.sh afhq-dataset
```


## Sampling from trained models
You can download the model checkpoints from [Google Drive](https://drive.google.com/drive/folders/18aX0pU2rE8bnBAT6ytAxc8yTpel2aezg?usp=drive_link) and save them in the `dbfs/checkpoint` directory.

See `dbfs/dbfs_{DATASET}_sample.ipynb` for sampling from the trained models.


## Training from scratch
We train DBFS with single or multi A6000 GPUs for each dataset.

You can also adjust the `--batch_dim` and `--nproc-per-node` options according to your local resources.

### EMNIST ⇆ MNIST
#### For Single-GPU
```
CUDA_VISIBLE_DEVICES=0 python dbfs_mnist.py
```
#### For Multi-GPU
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 dbfs_mnist.py
```

### AFHQ-64 Wild ⇆ Cat
#### For Multi-GPU
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 dbfs_afhq.py
```

The running histories are available on [Weights & Biases](https://wandb.ai/bwpark99/dbfs-publication/workspace?nw=nwuserbwpark99) for reproducibility.

# Reference 
If you found our work useful for your research, please consider citing our work.

```
@article{park2024stochastic,
  title={Stochastic Optimal Control for Diffusion Bridges in Function Spaces},
  author={Park, Byoungwoo and Choi, Jungwon and Lim, Sungbin and Lee, Juho},
  journal={arXiv preprint arXiv:2405.20630},
  year={2024}
}
```

# Acknowledgements
Our code builds upon an outstanding open source projects and papers:
* [Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling](https://arxiv.org/abs/2304.00917).
* [Score-based Generative Modeling through Stochastic Evolution Equations in Hilbert Spaces](https://openreview.net/pdf?id=GrElRvXnEj).