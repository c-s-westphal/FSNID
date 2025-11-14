# FSNID
# Feature Selection for Network Intrusion Detection

This repository contains the code used to generate the results presented in the paper:

> **Feature Selection for Network Intrusion Detection**  
> *[Charles Westphal](https://c-s-westphal.github.io/), et al.*
> In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (SIGKDD 2025}
> [Link to the paper](https://dl.acm.org/doi/10.1145/3690624.3709339)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Citation](#citation)

## Introduction

In this repo, we publish the code used to create the results in Feature Selection for Network Intrusion Detection (FSNID). FSNID, is an information-theoretic filter method that sequentially eliminates features that fail to transfer entropy to the attack vector, as shown in the following schematic.
 
 ![2ndprojchematic2](https://github.com/user-attachments/assets/ee47cafe-36ac-4ec1-930a-f55d135d0d57)

This in turn led to us achieving the following main results:

![2ndprojbar](https://github.com/user-attachments/assets/96225887-6e96-425f-a863-4ef18983bb93)


## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/c-s-westphal/FSNID.git
   cd FSNID

2. **Create Virtual Environment:**

   ```bash
   python3 -m FSNID_venv venv
   source FSNID_venv/bin/activate

3. **Install Required Packages:**

   ```bash
   pip install -r requirements.txt  


## Usage
Run the feature selection and classification script using:

    ```bash
    python main.py --nme DATASET_NAME --selection_method METHOD --model_type MODEL

### Arguments

- `--nme`: Name of the dataset to use (default: `BOTIOT`).
- `--selection_method`: Feature selection method to use. Choices are `fsnid`, `brown`, `firefly`, `lasso`, `pi` (default: `fsnid`).
- `--model_type`: Type of model to evaluate the features with. Only FSNID is designed to be used with all four, other methods should be left to default to MLP. Choices are `MLP`, `LSTM`, `TCN`, `GRU` (default: `MLP`).

## Datasets

Due to GitHub's file size limitations, the full datasets are not included in this repository. However, the first 5000 rows of the **BOTIOT** dataset are provided in the `/data` directory to demonstrate the required format.

For the complete datasets, please visit:

- **BOT-IoT Dataset:** [Download Link](https://research.unsw.edu.au/projects/bot-iot-dataset)
- **TON-IoT Dataset:** [Download Link](https://research.unsw.edu.au/projects/toniot-datasets)
- **NSL-KDD Dataset:** [Download Link](https://www.unb.ca/cic/datasets/nsl.html)
- **CIC-DDoS2019 Dataset:** [Download Link](https://www.unb.ca/cic/datasets/ddos-2019.html)
- **UNSW-NB15 Dataset:** [Download Link](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **CIC-IDS2017 Dataset:** [Download Link](https://www.unb.ca/cic/datasets/ids-2017.html)


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{WHM25:feature,
author = {Westphal, Charles and Hailes, Stephen and Musolesi, Mirco},
title = {{Feature Selection for Network Intrusion Detection}}, 
year = {2025}, 
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (SIGKDD 2025}, 
location = {Toronto, Canada}
}

