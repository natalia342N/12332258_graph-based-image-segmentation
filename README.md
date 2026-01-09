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


# Assignment 2 Hacking 

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

## Baseline U-Net Performance

| Metric          | Value |
|-----------------|------:|
| Foreground IoU  | 0.543 |
| mIoU            | 0.600 |
| Dice Score      | 0.704 |
| Pixel Accuracy  | 0.756 |
| Validation Loss | 0.5199 |

These values are computed on the **held-out validation split**
(10% of the `trainval` subset) and correspond to the metrics obtained in the
**final epoch** of the baseline implementation (`baseline/train_unet.py`).

These values serve as the **baseline** for further experiments in Assignment 2 (Hacking).

## Hacked Experiment (E2) – U-Net with Data Augmentation
We evaluate our models using two standard segmentation metrics:

1. IoU (Intersection-over-Union) on the foreground class.
2. Dice Score.

| Model            | Epochs | Batch Size | Augmentation        | IoU_fg | Dice  | Val Loss |
|------------------|:------:|:----------:|---------------------|:------:|:-----:|:--------:|
| Baseline U-Net   |   3    |     4      | None                | 0.543  | 0.704 | 0.5199   |
| Hacked U-Net     |  10    |     8      | Brightness + Noise  | 0.587  | 0.740 | 0.4323   |


The hacked model clearly outperforms the baseline on both of our primary
metrics (foreground IoU and Dice score) and also achieves a lower validation
loss.

## Metric Targets vs. Achieved

| Metric | Target | Achieved (E1) | Achieved (E2) | Expected (E3) |
|--------|:------:|:-------------:|:-------------:|:-------------:|
| IoU_fg | ≥ 0.55 | 0.543         | **0.587**     | **0.58–0.62** |
| Dice   | ≥ 0.70 | 0.704         | **0.740**     | **0.74–0.78** |
| Val Loss | –    | 0.5199        | **0.4323**    | TBD           |



### Testing Approach
Following the assignment requirements, we implement small testing cases to:

- Verify Oxford Pets dataset loads with correct shapes and value ranges
- Confirm image normalization and augmentation produce valid tensors  
- Test IoU and Dice calculations with known ground truths
- Ensure U-Net architecture accepts our input dimensions


## Reproducing the Experiments (of Assignment 2)

All experiments were run in a Python virtual environment (`venv`).

### 1. Clone and set up the environment

```bash
git clone https://github.com/natalia342N/12332258_graph-based-image-segmentation.git
cd 12332258_graph-based-image-segmentation

python3 -m venv adl_env
source adl_env/bin/activate 

pip install -r requirements.txt

cd tests
python3 test_baseline.py
python3 test_metrics.py
cd ..

run the baseline training:

python3 -m baseline.train_unet \
    --epochs 3 \
    --batch-size 4 \
    --img-size 256

run the hacked training:

python3 -m baseline.train_unet_hacked \
    --epochs 10 \
    --batch-size 8 \
    --img-size 256


```

Tested with Python 3.11 on macOS. 

## Project Timeline Update 1 

| Task                        | Estimated | Actual | Status |
|----------------------------|-----------|--------|--------|
| Dataset selection & prep   | 8h        | 6h     | Completed |
| Literature review          | 16h       | 10h    | Completed |
| Model implementation       | 24h       | 10h    | In progress |
| Training & fine-tuning     | 16h       | 10h    | In progress |
| Graph-based method (GCN)   | –         | 2h     | TO DO |
| Application & visualization| 16h       | 0h     | TO DO |
| Report writing             | 24h       | 0h     | TO DO |
| Presentation preparation   | 12h       | 0h     | TO DO |



# Assignment 3 Deliver

### Relation to Prior Work

This project is inspired by recent work on graph-based image segmentation. In particular, Jung et al. (2023) propose a superpixel-based GCN that performs segmentation by modeling spatial relationships between image regions. We adopt a similar superpixel graph construction strategy in Experiment E3, using GraphSAGE for node classification.

Furthermore, Singh et al. (2025) demonstrate that combining convolutional feature extractors with graph-based reasoning can improve segmentation quality. Motivated by this idea, we extend our graph-only model with a hybrid CNN–GNN approach in Experiment E4, where intermediate U-Net encoder features are pooled over superpixels and refined using a GNN.

Rather than reproducing large-scale architectures from the literature, our goal is to implement these concepts in a lightweight and reproducible form, allowing a clear comparison between CNN-only, GNN-only, and hybrid models.


## Experiments Summary

| Model / Experiment | Description | IoU_fg | Dice |
|---|---|---:|---:|
| E1 Baseline U-Net | 3 epochs, no augmentation | 0.543 | 0.704 |
| E2 Hacked U-Net | 10 epochs, brightness + noise | 0.587 | 0.740 |
| E3 Superpixel GraphSAGE | SLIC superpixels + handcrafted node features | 0.597 | 0.730 |
| **E4 Hybrid CNN+GNN (final)** | U-Net encoder features pooled over superpixels + GraphSAGE | **0.691** | **0.806** |



### Feedback applied from Assignment 2
## Hyperparameter Sensitivity Analysis

To study the sensitivity of the proposed hybrid CNN–GNN model to training hyperparameters, we evaluated multiple configurations varying the learning rate, weight decay, number of superpixels, and GNN hidden dimensionality. Rather than relying on a single training setup, we explicitly compared three configurations and selected the best-performing one based on validation IoU. This analysis showed that moderate learning rates combined with weight decay improved generalization, while excessively large superpixel graphs did not yield further gains.


## Hyperparameter Exploration (E4)

| Experiment | Learning Rate | Weight Decay | Hidden Dim | Num Layers | Best Epoch | Best Val Node Acc |
|---|---:|---:|---:|---:|---:|---:|
| **E4-A** | 1e-3 | 0.0 | 64 | 2 | 30 | **0.8000** |
| E4-B | 5e-4 | 1e-4 | 64 | 2 | 30 | 0.7859 |
| E4-C | 5e-4 | 1e-4 | 32 | 2 | 29 | 0.7765 |



To evaluate the robustness of the hybrid CNN–GNN model (E4), we trained three configurations with different learning rates, weight decay values, and GNN hidden dimensions. The best performance was achieved by **E4-A**, which used a learning rate of 1e-3 and a hidden dimension of 64, reaching a validation node accuracy of 0.842. Lower learning rates, added weight decay, or reduced model capacity did not improve performance in this setting. Based on these results, E4-A was selected as the final configuration for evaluation and demonstration.


## TensorBoard Tracking

Training/validation loss and node accuracy are logged to TensorBoard for each experiment:

```bash
tensorboard --logdir runs


We implemented early stopping (patience=5, min_delta=5e-4) based on validation node accuracy and save both the best checkpoint and the last checkpoint for each run.
```


## Metric Targets vs. Achieved

| Metric | Target | Achieved (E1) | Achieved (E2) | Achieved (E3) | Achieved (E4) |
|---|:---:|:---:|:---:|:---:|:---:|
| IoU_fg | ≥ 0.55 | 0.543 | **0.587** | **0.597** | **0.691** |
| Dice | ≥ 0.70 | 0.704 | **0.740** | **0.730** | **0.806** |


### Combined Experiment Reproduction Script

```bash
rm -rf data/pets_graphs
rm -rf data/pets_graphs_unetfeat
rm -rf runs

python baseline/train_unet_e1.py
python baseline/train_unet_e2.py

python -m graph.preprocess.pets_preprocessing_e3
python -m graph.train.train_pets_GNN \
  --config configs/e3.yaml \
  --device cpu
python -m graph.eval.eval_e3 --config configs/e3.yaml --device cpu


python -m graph.preprocess.pets_preprocessing_e4
python -m graph.train.train_pets_GNN --config configs/e4_a.yaml
python -m graph.train.train_pets_GNN --config configs/e4_b.yaml
python -m graph.train.train_pets_GNN --config configs/e4_c.yaml

python -m graph.eval.eval_e4 \
  --device cpu \
  --graphs_dir data/pets_graphs_unetfeat \
  --ckpt runs/E4-A/best_gnn.pth
python -m graph.eval.eval_e4 --device cpu --graphs_dir data/pets_graphs_unetfeat --ckpt runs/E4-B/best_gnn.pth
python -m graph.eval.eval_e4 --device cpu --graphs_dir data/pets_graphs_unetfeat --ckpt runs/E4-C/best_gnn.pth


pytest -q
```



