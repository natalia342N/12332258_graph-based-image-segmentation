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

..

## Project Timeline 

..