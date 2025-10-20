# Kidney Tumor Detection in CT Images using Advanced Artificial Intelligence Techniques

**Author:** Joaquín Mir Macías  
**Supervisor:** Meritxell Riera i Marín  
**Collaborating Entity:** Sycai Medical – ICAI  


## Overview

Renal cancer is a highly prevalent disease whose early detection through **advanced artificial intelligence** techniques can significantly improve patient prognosis.  
This project presents a **comparative evaluation of several Deep Learning and Computer Vision models** for the **automatic and precise segmentation of renal tumors** in **computed tomography (CT)** images.

Public datasets **KiTS19**, **KiTS21**, and **KiTS23** are used to analyze multiple **convolutional architectures**, focusing on both **2D and 3D segmentation** strategies.  
The models compared include **U-Net**, **nnU-Net**, **FCN**, **DeepLab**, **MONAI Auto3DSeg**, and a training framework with **Stochastic Gradient Descent with Restarts (SGDR)**.


## Objective

To experimentally evaluate and compare the most relevant **CNN architectures** for kidney tumor segmentation, identifying the model that offers the best trade-off between **accuracy**, **robustness**, and **reliability**, considering volumetric precision and uncertainty estimation.


## Models Evaluated

| **Model** | **Description** | **Key Features** | **Tumor Dice** |
|------------|-----------------|------------------|----------------|
| **U-Net (2D/3D)** | Classical encoder–decoder architecture. The 3D version leverages volumetric coherence. | Lightweight, fast, baseline performance. | 0.65–0.70 |
| **FCN** | Fully Convolutional Network for semantic segmentation. | Simplified end-to-end design, reference baseline. | 0.66 |
| **DeepLab** | Multi-scale encoder–decoder with atrous convolutions. | Captures contextual information at different scales. | 0.73 |
| **nnU-Net (2D, 3D, Cascade)** | Self-configuring framework adapting preprocessing, patch size, and augmentations. | Automates training and inference setup; strong robustness. | 0.85 |
| **MONAI Auto3DSeg** | AutoML pipeline from MONAI that trains and ensembles models such as SegResNet, DynUNet, and Swin-UNETR. | Automated model selection and ensembling; best overall accuracy. | 0.87 |
| **SGDR** | Training strategy employing Stochastic Gradient Descent with Restarts. | Improves convergence and generalization of CNNs. | — |


## Datasets

- **KiTS19**, **KiTS21**, **KiTS23**  
  Publicly available datasets containing annotated CT scans for **kidney and tumor segmentation**.  
  Each case includes **3D volumes** with expert-labeled masks for training and evaluation.


## Metrics

Model performance is assessed using the following metrics:

- **Dice Coefficient (Dice)** – measures volumetric overlap  
- **Surface Dice (ω)** – evaluates boundary accuracy with tolerance  
- **Hausdorff Distance (HD95)** – assesses the 95th-percentile surface distance  
