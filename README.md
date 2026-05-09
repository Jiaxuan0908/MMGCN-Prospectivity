# 3D Mineral Prospectivity Modeling via MMGCN
This repository contains the official source code for our paper on 3D mineral prospectivity modeling using the Multiscale Mixture of Experts Graph Convolutional Network (MMGCN).

## Environment Setup
The required Python packages are listed in the `requirements.txt` file. 

**Important Note:** Core deep learning libraries such as PyTorch and Deep Graph Library (DGL) usually require specific versions matching your local CUDA drivers. Therefore, please do **not** simply run `pip install -r requirements.txt`. We highly recommend installing PyTorch and DGL manually from their official websites according to your specific hardware environment, and then installing the remaining regular packages.

## Data Availability Statement
Due to strict confidentiality agreements regarding the deep geological exploration data, the `data` folder provided in this repository only contains structural format demonstrations (dummy data). 

These sample files are provided solely to illustrate the input data formats and structures required by the model. **The codes cannot be executed directly with these sample files to reproduce the results.** For detailed descriptions of the original geological and geochemical datasets, please refer to the corresponding sections in our published paper.

## Code Structure
This repository provides the core implementation of our framework. The specific functions of the uploaded scripts are as follows:

* **`Data_loader.py`**: Scripts for loading raw geological data, normalizing features, and constructing the multiscale heterogeneous graphs.
* **`MMGCN.py`**: The core implementation of the Multiscale Mixture of Experts Graph Convolutional Network architecture, including the dual-scale GCN and the cross-modal expert fusion mechanism.
* **`Prediction.py`**: The execution script for conducting spatial inference and generating the final 3D mineral prospectivity models.
* **`Stability_test.py`**: Scripts designed to run independent trials under varying random seeds to evaluate model robustness and calculate quantitative uncertainty (Standard Deviation and Information Entropy).
* **`Ablation.py`**: Scripts for performing specific ablation studies to validate the necessity of heterogeneous data fusion and internal network modules.
