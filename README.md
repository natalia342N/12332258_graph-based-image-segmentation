# 12332258_graph-based-image-segmentation
194.077 Applied Deep Learning WS25 Project Repository


## Introduction

This repository contains the code and resources for the project "Graph-Based Image Segmentation" as part of the 194.077 Applied Deep Learning course in Winter Semester 2025.

Image segmentation has recently been widely explored in deep learning, especially through graph-based neural networks. While plain Convolutional Neural Networks (CNNs) provide a strong baseline, studies show that hybrid CNN-Graph Neural Network (GNN) architectures can futher improve segmentation quality and accuracy. In this project, we build on recent advances of a graph-based method with superpixel boundary structure recognition. This topic through the proposed approach will be evaluated on the benchmark dataset Pascal VOC. 


The project will be conducted with an approach of "Bring Your Own Method". We are implementing and trying to improve based on publicly available architectures, comparing CNN and GNN based methods. 

## Scientific Paper References

- [1] H. Jung, S.Y. Park, S. Yang, and J. Kim, Superpixel-based Graph Convolutional Network for Semantic Segmentation (Seoul National University, 2023), available at: https://github.com/HoinJung/SuperpixelGCN-Segmentation [accessed 13 October 2025].

- [2] Singh, A., Van de Ven, P., Eising, C., and Denny, P., Image Segmentation: Inducing Graph-Based Learning. arXiv:2501.03765, version 2 (19 January 2025). Available at: arXiv.org (accessed 13 October 2025)

## Reseach Questions

The CNN and GNN based method will be examined as a baseline model while the proposed Superpixel-based GNN will aim to improve the segmentation quality by replacing the CNN encoder-decoder with a graph built directly on superpixels. The questions we want to answer in this project are:

1. Is this approach possible to implement with my skillset and time frame?
2. What is a current performance baseline for image segmentation on Pascal VOC?
3. How does the GNN based method compare to the CNN based method in terms of segmentation and further how does the superpixel-based GNN perform compared to both?
4. Which graph construction method is most suitable for this task but also which method is applicable with limited GPU resources?
5. ..

## Dataset 

Among popular image segmentation datasets, the Oxford-IIIT Pet Dataset was selected for this project. It provides annotated images of cats and dogs from 37 different breeds. The dataset contains 7349 images, divided into training/validation and test splits.

The dataset was loaded using torchvision.datasets.OxfordIIITPet, which automatically downloads both images and their corresponding segmentation masks. In the accompanying Jupyter notebook, an initial dataset inspection and visualization was performed (without training) to verify that the dataset size, accessibility, and memory requirements do not pose any issues for this project.

Each sample consists of an RGB image and a corresponding segmentation mask of equal spatial dimensions, containing three pixel classes: background, foreground (the animal), and boundary, along with their respective pixel counts.

## Project Timeline 

| Task | Description | Estimated Time |
| ----- | ------------ | --------------- |
| Dataset selection and preparation | Download the Oxford-IIIT Pet dataset, inspect samples and segmentation masks, verify accessibility and memory usage. | 8 hours |
| Literature Review | Review papers on CNN–GNN hybrid segmentation models and define the model architecture (baseline U-Net and graph module). Verify GitHub repositories for implementation details. Try to reproduce key results (if possible on related datasets) | 16 hours |
| Model implementation | Implement the network in PyTorch with all the functionality. Adapt it for Oxford-IIIT Pet Dataset | 24 hours |
| Training and fine-tuning | Train and optimize the model, tune hyperparameters. | 16 hours |
| Application and visualization | Build an application to visualize predictions vs. ground-truth masks and compute evaluation metrics. | 16 hours |
| Report writing | Write the report. | 24 hours |
| Presentation preparation | Prepare slides and visuals for the project presentation. | 12 hours |

**Total estimated:** ≈ allocate minimum 80 hours within the next 7 weeks ≈ 5 hours 2 times per week


### Project Inspiration 

The motivation for choosing this project did not come from a random glance at the list of topics, but a real-life situation. While driving a modern car equipped with a road sign detection system, I once found myself wondering — how is this actually possible, and what mechanism lies behind it?

At first, I assumed the system might rely on a GPS database that stores information about road signs and updates them on the driver’s display as the car moves. However, such systems can now update speed limits almost instantly after passing a sign — or even slightly before. This raised the question: how does such automation work from the ground?

A brief investigation revealed that modern driver-assistance systems make extensive use of deep learning approaches, particularly image segmentation. This discovery inspired the idea for this project — to explore how deep learning techniques can enable machines to interpret visual scenes with pixel-level accuracy, just like human perception on the road.