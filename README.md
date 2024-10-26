# Mammo-DETR

This repository contains the code for the paper "Understanding differences in applying DETR to natural and medical images".

## Overview

Mammo-DETR is a project that evaluates the applicability of transformer-based object detection models, specifically DETR (Detection Transformer), to medical imaging data, with a focus on screening mammography. The project aims to understand how design choices optimized for natural images perform when applied to the unique challenges presented by medical imaging data.

## Key Findings

Our research reveals that:

1. Common design choices from natural image domain often do not improve, and sometimes impair, object detection performance in medical imaging.
2. Simpler and shallower architectures often achieve equal or superior results in medical imaging tasks.
3. The adaptation of transformer models for medical imaging data requires a reevaluation of standard practices.

## Repository Structure

- `sample_data/`: Contains sample mammography images for testing
- `sample_output/`: Includes example outputs from the model
- `src/`: Source code for the Mammo-DETR model
- `requirements.txt`: List of Python dependencies
- `run.sh`: Shell script for local execution
- `models`: We released the [model Checkpoints](https://drive.google.com/drive/folders/1h-VD5mr1xfa4pvzQCpox8y7u-R84Hkpu?usp=drive_link)


## Installation

### Requirements
This is adopted from deformable DETR [here](https://github.com/fundamentalvision/Deformable-DETR/tree/main )
* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd src/modeling/def_detr/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Run the inference

```bash

./run.sh

```
## License

This repository is licensed under the terms of the GNU AGPLv3 license.

## Citation
If you use this code in your research, please cite our paper:

@article{xu2024understanding,
  title={Understanding differences in applying DETR to natural and medical images},
  author={Xu, Yanqi and [Other Authors]},
  journal={arXiv preprint arXiv:2405.17677},
  year={2024}
}
