# 12332258_graph-based-image-segmentation
194.077 Applied Deep Learning WS25 Project Repository


## Introduction

This repository contains the code and resources for the project Graph-Based Image Segmentation, developed as part of the 194.077 Applied Deep Learning course in the Winter Semester 2025.

Image segmentation has recently been extensively explored in deep learning, particularly through graph-based neural networks. While conventional Convolutional Neural Networks (CNNs) provide a strong baseline, studies show that hybrid CNN–Graph Neural Network (GNN) architectures can further improve segmentation quality and accuracy. In this project, we build upon recent advances in graph-based methods that incorporate superpixel boundary structure recognition. The proposed approach will be evaluated on the benchmark dataset Oxford-IIIT Pets.

The project follows a “Bring Your Own Method” approach. We implement and aim to improve upon publicly available architectures, comparing CNN- and GNN-based methods.

## Scientific Paper References

- [1] H. Jung, S.Y. Park, S. Yang, and J. Kim, Superpixel-based Graph Convolutional Network for Semantic Segmentation (Seoul National University, 2023), code available at: https://github.com/HoinJung/SuperpixelGCN-Segmentation [accessed 13 October 2025].

- [2] A. Singh, P. Van de Ven, C. Eising, and P. Denny, Image Segmentation: Inducing Graph-Based Learning, arXiv:2501.03765, version 2 (19 January 2025), paper available at: arXiv.org, code available at: https://github.com/aryan-at-ul/Electronic-Imaging-2025-paper-4492 [accessed 13 October 2025].


## Reseach Questions

The CNN- and GNN-based methods will be examined as baseline models, while the proposed superpixel-based GNN aims to improve segmentation quality by replacing the CNN encoder–decoder with a graph constructed directly on superpixels.

The main research questions for this project are as follows:

1. Is this approach feasible to implement within the given skill set and time frame?
2. What is the current performance baseline for image segmentation on the Oxford Pets dataset?
3. How does the GNN-based method compare to the CNN-based method in terms of segmentation performance, and how does the superpixel-based GNN perform relative to both?
4. Which graph construction method is most suitable for this task, and which approach is most practical given limited GPU resources?         


## Dataset 

Among the popular image segmentation datasets, the Oxford-IIIT Pet Dataset was selected for this project. It provides annotated images of cats and dogs from 37 different breeds. The dataset contains 7,349 images, divided into training, validation, and test splits.

The dataset was loaded using torchvision.datasets.OxfordIIITPet, which automatically downloads both the images and their corresponding segmentation masks. In the accompanying Jupyter notebook, an initial dataset inspection and visualization were performed (without training) to verify that the dataset size, accessibility, and memory requirements do not pose any issues for this project.

Each sample consists of an RGB image and a corresponding segmentation mask of equal spatial dimensions. The mask contains three pixel classes: background, foreground (the animal), and boundary, along with their respective pixel counts.

## Project Timeline 

| Task | Description | Estimated Time |
| ----- | ------------ | --------------- |
| Dataset selection and preparation | Download the Oxford-IIIT Pet dataset, inspect samples and segmentation masks, verify accessibility and memory usage. | 8 hours |
| Literature Review | Review papers on CNN–GNN hybrid segmentation models and define the model architecture (baseline U-Net and graph module). Verify GitHub repositories for implementation details. Try to reproduce key results (if possible on related datasets). | 16 hours |
| Model implementation | Implement the network in PyTorch with all the functionality. Adapt it for Oxford-III Pet Dataset. | 24 hours |
| Training and fine-tuning | Train and optimize the model, tune hyperparameters. | 16 hours |
| Application and visualization | Build an application to visualize predictions vs. ground-truth masks and compute evaluation metrics. | 16 hours |
| Report writing | Write the report. | 24 hours |
| Presentation preparation | Prepare slides and visuals for the project presentation. | 12 hours |

Total estimated effort: approximately 80 hours over the next 7 weeks (until submission 2), equivalent to two 5-hour sessions per week.

Progress so far: Initiated in weeks 0–2, with approximately 15 hours already completed.


### Project Inspiration 

The motivation for choosing this project did not arise from a random glance at a list of topics but from a real-life experience. While driving a modern car equipped with a road sign detection system, I found myself wondering—how is this actually possible, and what mechanism lies behind it?

Initially, I assumed that the system might rely on a GPS database storing information about road signs, updating them on the driver’s display as the car moves. However, I soon realized that these systems can update speed limits almost instantly after passing a sign—or even slightly before. This observation raised a deeper question: how does such automation truly work at the fundamental level?

A brief investigation revealed that modern driver-assistance systems make extensive use of deep learning, particularly image segmentation. This discovery inspired the idea for this project: to explore how deep learning techniques enable machines to interpret visual scenes with pixel-level precision—closely resembling human perception on the road.


### Assignment 2 Hacking 

## Baseline Experiment (E1) – U-Net, Binary Segmentation

For the initial baseline, a standard U-Net model was trained on the Oxford-IIIT Pet dataset with binary segmentation (pet vs. background).

Configuration:

- Input resolution: 256 × 256
- Output: 1 channel 
- Loss: BCEWithLogitsLoss
- Optimizer: Adam 
- Epochs: 3
- Batch size: 4
- Train/val split: 90% / 10% from the `trainval` split
- Data augmentation: none

**Results**

| Metric         | Value |
|----------------|------:|
| Foreground IoU | 0.619 |
| mIoU           | 0.655 |
| Dice           | 0.765 |
| Pixel accuracy | 0.794 |
| Val loss       | 0.445 |

These values serve as the **baseline** for further experiments in Assignment 2 (Hacking), where I aim to improve IoU and mIoU by increasing input resolution, adding data augmentation, and tuning training hyperparameters.
