# Hardware Resilience Properties of Text-Guided Image Classifiers [NeurIPS 2023] 

[Syed Talal Wasim](https://talalwasim.github.io),
[Kabila Haile Soboka](https://muzairkhattak.github.io/),
[Abdulrahman Mahmoud](https://ma3mool.github.io/),
[Salman Khan](https://salman-h-khan.github.io/),
[David Brooks](http://www.eecs.harvard.edu/~dbrooks/),
[Gu-Yeon Wei](https://seas.harvard.edu/person/gu-yeon-wei)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://github.com/TalalWasim/TextGuidedResilience)
<hr />

> **Abstract:**
>*This research paper presents a novel method to enhance the reliability of image classification models during deployment in the face of transient hardware errors. By utilizing enriched text embeddings derived from GPT-3 with question prompts per class and CLIP pretrained text encoder, we investigate their impact as an initialization for the classification layer. Our approach achieves a remarkable 4-14x increase in hardware reliability, across various architectures, while maintaining minimal accuracy drop compared to baseline PyTorch models. Furthermore, our method seamlessly integrates with any image classification backbone, showcases results across various architectures, decreases parameter and FLOPs overhead, and follows a consistent training recipe. This research offers a practical and efficient solution to bolster the robustness of image classification models against hardware failures, with potential implications for future studies in this domain. Our code and models will be publicly released.*

## Table of Contents
<!--ts-->
   * [News](#rocket-News)
   * [Overview](#overview)
   * [Visualization](#visualization-first-and-last-layer-spatio-temporal-modulator)
   * [Environment Setup](#environment-setup)
   * [Dataset Preparation](#dataset-preparation)
   * [Model Zoo](#model-zoo)
      * [Kinetics-400](#kinetics-400)
      * [Kinetics-600](#kinetics-600)
      * [Something-Something-v2](#something-something-v2)
      * [Diving-48](#diving-48)
      * [ActivityNet-v1.3](#activitynet-v13)
   * [Evaluation](#evaluation)
   * [Training](#training)
   * [Citation](#citation)
   * [Acknowledgements](#acknowledgements)
<!--te-->

## :rocket: News
* **(July 13, 2022)** 
  * Training and evaluation codes for Video-FocalNets, along with pretrained models are released.
<hr />


## Overview

<p align="center">
  <img alt="Overall Architecture" src="figs/overall_architecture.png" width="1200"/>
  <p align="center"><b>(a) The overall architecture of Video-FocalNets:</b> A four-stage architecture, with each stage comprising a patch embedding and a number of Video-FocalNet blocks. <b>(b) Single Video-FocalNet block:</b> Similar to the transformer blocks, we replace self-attention with Spatio-Temporal Focal Modulation.</p>
</p>
<hr />
<p align="center">
  <table>
  <tr>
    <td><img alt="Overall Architecture" src="figs/overview_focal_modulation.png" width="98%"></td>
    <td><img alt="Performance Comparison" src="figs/intro_plot.png" width="98%"></td>
  </tr>
  <tr>
    <td><p align="center"><b>The Spatio-Temporal Focal Modulation layer:</b> A spatio-temporal focal modulation block that independently models the spatial and temporal information.</p></td>
    <td><p align="center"><b>Comparison for Top-1 Accuracy vs GFlops/view on Kinetics-400.</b></p></td>
  </tr>
 </table>
</p>

## Visualization: First and Last layer Spatio-Temporal Modulator

<p align="center">
  <img alt="Visualization Cutting Apple" src="figs/vis/cutting_apple.png" width="900"/>
</p>

<p align="center">
  <img alt="Visualization Scuba Diving" src="figs/vis/scuba_diving.png" width="900"/>
</p>

<p align="center">
  <img alt="Visualization Threading Needle" src="figs/vis/threading_needle.png" width="900"/>
</p>

<p align="center">
  <img alt="Visualization Walking the Dog" src="figs/vis/walking_the_dog.png" width="900"/>
</p>

<p align="center">
  <img alt="Visualization Water Skiing" src="figs/vis/water_skiing.png" width="900"/>
</p>


## Environment Setup
Please follow [INSTALL.md](./INSTALL.md) for installation.

## Dataset Preparation

Please follow [DATA.md](./DATA.md) for data preparation.

## Model Zoo

### Kinetics-400

|       Model      |    Depth   | Dim | Kernels | Top-1 | Download |
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|
| Video-FocalNet-T |  [2,2,6,2] |  96 |  [3,5]  |  79.8 |   [ckpt](https://drive.google.com/file/d/1wsUjJbPVQd7pf-OocD9mVU8pak0gdBTP/view?usp=sharing)   |
| Video-FocalNet-S | [2,2,18,2] |  96 |  [3,5]  |  81.4 |   [ckpt](https://drive.google.com/file/d/1gO4_tluuoR4mn2bSQRNyy9_wFCnUSiQ0/view?usp=sharing)   |
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  83.6 |   [ckpt](https://drive.google.com/file/d/1tc1AKKmvHN7Hzxpd53QsBIMQZmLH8ozX/view?usp=drive_link)   |

### Kinetics-600

|       Model      |    Depth   | Dim | Kernels | Top-1 | Download |
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  86.7 |   [ckpt](https://drive.google.com/file/d/16u1dij3dde0KmaajiB5lAFy8FaRvQDmS/view?usp=sharing)   |

### Something-Something-v2

|       Model      |    Depth   | Dim | Kernels | Top-1 | Download |
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  71.1 |   [ckpt](https://drive.google.com/file/d/1MIPLjMVDmYEY5jmJs8pRRIj4gKNVqETg/view?usp=sharing)   |

### Diving-48

|       Model      |    Depth   | Dim | Kernels | Top-1 | Download |
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  90.8 |   [ckpt](https://drive.google.com/file/d/1MMZeDucN1cfC5MiTGIft8xNfo5358dA2/view?usp=sharing)   |

### ActivityNet-v1.3

|       Model      |    Depth   | Dim | Kernels | Top-1 | Download |
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  89.8 |   [ckpt](https://drive.google.com/file/d/1Zku86i9Ol1gabqBqf0h1vtL-_H5gglA3/view?usp=sharing)   |



## Evaluation

To evaluate pre-trained Video-FocalNets on your dataset:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py  --eval \
--cfg <config-file> --resume <checkpoint> \
--opts DATA.NUM_FRAMES 8 DATA.BATCH_SIZE 8 TEST.NUM_CLIP 4 TEST.NUM_CROP 3 DATA.ROOT path/to/root DATA.TRAIN_FILE train.csv DATA.VAL_FILE val.csv
```

For example, to evaluate the `Video-FocalNet-B` with a single GPU on Kinetics400:

```bash
python -m torch.distributed.launch --nproc_per_node 1  main.py  --eval \
--cfg configs/kinetics400/video_focalnet_base.yaml --resume video-focalnet_base_k400.pth \
--opts DATA.NUM_FRAMES 8 DATA.BATCH_SIZE 8 TEST.NUM_CLIP 4 TEST.NUM_CROP 3 DATA.ROOT path/to/root DATA.TRAIN_FILE train.csv DATA.VAL_FILE val.csv
```

Alternatively, the `DATA.ROOT`, `DATA.TRAIN_FILE`, and `DATA.VAL_FILE` paths can be set directly in the config files provided in the `configs` directory.
According to our experience and sanity checks, there is a reasonable random variation of about +/-0.3% top-1 accuracy when testing on different machines.

Additionally, the TRAIN.PRETRAINED_PATH can be set (either in the config file or bash script) to provide a pretrained model to initialize the weights. To initialize from the ImageNet-1K weights please refer to the [FocalNets](https://github.com/microsoft/FocalNet) repository and download the [FocalNet-T-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_tiny_srf.pth), [FocalNet-S-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_small_srf.pth) or [FocalNet-B-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_base_srf.pth) to initialize Video-FocalNet-T, Video-FocalNet-S or Video-FocalNet-B respectively. Alternatively, one of the provided pretrained Video-FocalNet models can also be utilized to initialize the weights.


## Training

To train a Video-FocalNet on a video dataset from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py \
--cfg <config-file> --batch-size <batch-size-per-gpu> --output <output-directory> \
--opts DATA.ROOT path/to/root DATA.TRAIN_FILE train.csv DATA.VAL_FILE val.csv
```

Alternatively, the `DATA.ROOT`, `DATA.TRAIN_FILE`, and `DATA.VAL_FILE` paths can be set directly in the config files provided in the `configs` directory. We also provide bash scripts to train Video-FocalNets on various datasets in the `scripts` directory.

Additionally, the TRAIN.PRETRAINED_PATH can be set (either in the config file or bash script) to provide a pretrained model to initialize the weights. To initialize from the ImageNet-1K weights please refer to the [FocalNets](https://github.com/microsoft/FocalNet) repository and download the [FocalNet-T-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_tiny_srf.pth), [FocalNet-S-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_small_srf.pth) or [FocalNet-B-SRF](https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_base_srf.pth) to initialize Video-FocalNet-T, Video-FocalNet-S or Video-FocalNet-B respectively. Alternatively, one of the provided pretrained Video-FocalNet models can also be utilized to initialize the weights.


## Citation
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@InProceedings{Wasim_2023_ICCV,
    author    = {Wasim, Syed Talal and Khattak, Muhammad Uzair and Naseer, Muzammal and Khan, Salman and Shah, Mubarak and Khan, Fahad Shahbaz},
    title     = {Video-FocalNets: Spatio-Temporal Focal Modulation for Video Action Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at syed.wasim@mbzuai.ac.ae or uzair.khattak@mbzuai.ac.ae.

## Acknowledgements
Our code is based on [FocalNets](https://github.com/microsoft/FocalNet), [XCLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP) and [UniFormer](https://github.com/Sense-X/UniFormer) repositories. We thank the authors for releasing their code. If you use our model, please consider citing these works as well.
