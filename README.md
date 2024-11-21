# FSNID
# Feature Selection for Network Intrusion Detection

This repository contains the code used to generate the results presented in the paper:

> **Feature Selection for Network Intrusion Detection**  
> *[Charles Westphal], et al.*  
> [arXiv:2411.11603](https://arxiv.org/abs/2411.11603)
> To appear in KDD'25.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Citation](#citation)

## Introduction

In this repo we publish the code used to create the results in Feature Selection for Network Intrusion Detection (FSNID). FSNID, is an information-theoretic filter method that sequentially selects features that transfer entropy to the target, as shown in the following schematic. [2ndprojchematic2.pdf](https://github.com/user-attachments/files/17850746/2ndprojchematic2.pdf) This in turn lead to us achieving the following main results: 
[2ndprojbar.pdf](https://github.com/user-attachments/files/17850751/2ndprojbar.pdf)



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
- `--model_type`: Type of model to evaluate the features with. Choices are `MLP`, `LSTM`, `TCN`, `GRU` (default: `MLP`).

## Datasets

Due to GitHub's file size limitations, the full datasets are not included in this repository. However, the first 5000 rows of the **BOTIOT** dataset are provided in the `/data` directory to demonstrate the required data format.

For the complete datasets, please visit:

- **BOT-IoT Dataset:** [Download Link](https://research.unsw.edu.au/projects/bot-iot-dataset)
- **TON-IoT Dataset:** [Download Link](https://research.unsw.edu.au/projects/toniot-datasets)
- **NSL-KDD Dataset:** [Download Link](https://www.unb.ca/cic/datasets/nsl.html)
- **CIC-DDoS2019 Dataset:** [Download Link](https://www.unb.ca/cic/datasets/ddos-2019.html)
- **UNSW-NB15 Dataset:** [Download Link](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **CIC-IDS2017 Dataset:** [Download Link](https://www.unb.ca/cic/datasets/ids-2017.html)


## Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@misc{westphal2024featureselectionnetworkintrusion,
      title={Feature Selection for Network Intrusion Detection}, 
      author={Charles Westphal and Stephen Hailes and Mirco Musolesi},
      year={2024},
      eprint={2411.11603},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.11603}, 
}
