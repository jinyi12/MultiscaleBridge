<h1 align='center'>Stochastic Optimal Control for Diffusion Bridges in Function Spaces (DBFS)</h1>
<div align="center">
  <a href="https://bw-park.github.io/" target="_blank">Byoungwoo Park</a><sup>1</sup>&ensp;<b>&middot;</b>&ensp;
  <a href="https://jungwon-choi.github.io/" target="_blank">Jungwon Choi</a><sup>1</sup>&ensp;<b>&middot;</b>&ensp;
  <a href="https://www.sungbin-lim.net/" target="_blank">Sungbin Lim</a><sup>2,3</sup>&ensp;<b>&middot;</b>&ensp;
  <a href="https://juho-lee.github.io/" target="_blank">Juho Lee</a><sup>1</sup><br>
  <sup>1</sup>KAIST &emsp; <sup>2</sup>Korea University &emsp; <sup>3</sup>LG AI Research<br>
</div>
<br>
<div align="center">
[![arxiv](https://img.shields.io/badge/NeurIPS2024-blue)](https://arxiv.org/abs/2405.20630)
</div>

## Examples
| pi_0 ⇆ pi_T | Results (left: pi_0 → pi_T, right: pi_0 ← pi_T) |
|-------------------------|-------------------------|
| EMNIST ⇆ MNIST (32x32) | <p float="left"> <img src="./assets/emnist2mnist_32.gif" alt="drawing" width="180"/>  <img src="./assets/mnist2emnist_32.gif" alt="drawing" width="180"/> </p>  |
| EMNIST ⇆ MNIST (64x64) | <p float="left"> <img src="./assets/emnist2mnist_64.gif" alt="drawing" width="180"/>  <img src="./assets/mnist2emnist_64.gif" alt="drawing" width="180"/> </p>  |
| EMNIST ⇆ MNIST (128x128) | <p float="left"> <img src="./assets/emnist2mnist_128.gif" alt="drawing" width="180"/>  <img src="./assets/mnist2emnist_128.gif" alt="drawing" width="180"/> </p>  |
| AFHQ-64 Wild ⇆ Cat (64x64) | <p float="left"> <img src="./assets/wild2cat_64.gif" alt="drawing" width="180"/>  <img src="./assets/cat2wild_64.gif" alt="drawing" width="180"/> </p> | 
| AFHQ-64 Wild ⇆ Cat (128x128) | <p float="left"> <img src="./assets/wild2cat_128.gif" alt="drawing" width="180"/>  <img src="./assets/cat2wild_128.gif" alt="drawing" width="180"/> </p> | 

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

## Training
```
python dbfs_mnist.py
```

```
python dbfs_afhq.py
```

## Evaluation





# Acknowledgements
Our code builds upon an outstanding open source project and paper:
* [Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling](https://arxiv.org/abs/2304.00917).