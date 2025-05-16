# Wafer Scratch Detection

This repository presents a complete machine learning and deep learning pipeline for detecting scratch defects on semiconductor wafers. The project addresses a real-world industry problem where elongated clusters of faulty dies, known as "scratches," must be identified and marked ("inked") to maintain die quality.

## 📌 Problem Description

Given a wafer map from a specific test operation, the goal is to classify each die as either part of a scratch or not. Scratches are sequences of dies (both bad and good) suspected to be affected by physical defects on the wafer. The detection must also consider wafers with **low yield**, where scratch detection should be skipped to avoid unnecessary inking.


## 🧠 Approaches Implemented

### 1. Classical ML

* **Feature Engineering**:

  * Normalized die coordinates (`NormX`, `NormY`)
  * Distance from center
  * Number of faulty neighbors
* **Models**:

  * Random Forest
  * XGBoost
  * LightGBM
* **Imbalance Handling**:

  * SMOTE oversampling
  * Class weight balancing
* **Threshold Tuning**:

  * Precision-recall-based threshold optimization

### 2. Deep Learning

* **Model**: U-Net with ResNet18 encoder
* **Input**: 3-channel wafer tensor (`DieX`, `DieY`, `IsGoodDie`)
* **Output**: Scratch segmentation mask
* **Loss**: BCE with `pos_weight`, tested with Focal Loss
* **Augmentations**: (optionally added)

## 📈 Results Summary

* Best classical ML model: XGBoost + SMOTE with F1 \~ 0.67
* Best DL model (U-Net + ResNet18): F1 \~ **0.83** on full validation set

