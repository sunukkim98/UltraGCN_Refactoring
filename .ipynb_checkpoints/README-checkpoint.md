# UltraGCN_Refactoring
A refactored implementation of the UltraGCN model

## UltraGCN

UltraGCN is an ultra-simplified formulation of graph convolutional networks for collaborative filtering. This repo provides the official open-source implementation of our paper: 

+ Kelong Mao, Jieming Zhu, Xi Xiao, Biao Lu, Zhaowei Wang, Xiuqiang He. [UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation](https://arxiv.org/pdf/2110.15114.pdf), in CIKM 2021.

## Model Overview

Graph Convolutional Networks (GCN) have been widely used for collaborative filtering. GCN models allow to capture higher-order  connections between users and items through its recursive message passing mechanism to aggregate neighborhood information. However, this message passing mechanism largely slows down the convergence of GCNs, especially when mini-batch sub-graph sampling is applied on large graphs. [LightGCN](https://arxiv.org/abs/2002.02126) reduces GCN models by removing feature transformations and nonlinear activations. In our work, UltraGCN was developed as an ultra-simplified formulation of GCNs, which skips explicit message passing and instead approximates infinite-layer graph convolutions using a constraint loss.

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/reczoo/RecZoo@main/matching/gnn/UltraGCN/img/ultragcn.png" width="500" alt="UltraGCN model"/>
</div>

## Environments

This project requires Python 3.7.6. It is recommended to use a virtual environment.

### Setting up the environment

1. **Create a virtual environment with Python 3.7.6:**
    ```bash
    conda create -n myenv python=3.7.6
    ```
2. **Install required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    Use `requriements.txt` to install all dependencies.

## Project Structure

## Results