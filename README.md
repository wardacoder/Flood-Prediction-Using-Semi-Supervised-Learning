# 🌊 Flood Prediction Using Semi-Supervised Learning

Floods are among the most devastating natural disasters in Bangladesh, with recurring events causing massive loss of life, infrastructure, and livelihoods. While real-time sensor networks are expanding, the availability of **labeled flood data** remains limited due to high annotation costs and logistical constraints.

This project introduces a robust semi-supervised learning framework centered on the **K-Nearest Neighbors (KNN)** algorithm, enhanced through **rigorous preprocessing**, **selective pseudo-labeling**, and a **three-phase training pipeline**. While KNN is typically considered simple, this work demonstrates how **thoughtful model design** and **data-centric strategies** can deliver **near-perfect performance**. The final model achieved a test **F1-score of 0.9986 (99.86%)**, outperforming more complex methods and setting a new **benchmark** for this widely used dataset. 

---

## 📌 Overview

- **Dataset**: 20,500+ records from Bangladesh (approx. 78% unlabeled)
- **Goal**: Robust and balanced prediction of flood and non-flood events in a data-scarce environment, with special emphasis on class-wise performance using test F1-score.
- **Method**: Semi-supervised learning with KNN and high-confidence pseudo-labeling

---

## ⚙️ Methodology

0. **Data Preprocessing**  
   Applied extensive preprocessing to improve signal quality and reduce noise:  
   - Removed non-predictive and redundant features  
   - Encoded categorical variables  
   - Visualized and removed outliers  
   - Analyzed feature distributions (e.g., skewness)  
   - Applied Min-Max scaling for skewed features and Standard scaling for normally distributed ones  
   - Evaluated model performance on both original and SMOTE-balanced datasets

1. **Initial Supervised Training**  
   Trained a base KNN model using 3,432 labeled samples after preprocessing.

2. **Selective Pseudo-Labeling**  
   Used the trained model to generate pseudo-labels for the unlabeled data (≈78%).  
   Only predictions with **≥95% confidence** were retained for reliability.

3. **Final Semi-Supervised Training**  
   Combined the original labeled data with high-confidence pseudo-labeled data (total: 13,910 records) to **retrain** the model. Tuned hyperparameters (neighbors, metric, weights) using GridSearchCV for optimal performance

---

## 📊 Results

| Metric       | Score       |
|--------------|-------------|
| **F1-score** | **0.9986**  |
| Precision    | 1.00        |
| Recall       | 1.00        |
| Accuracy     | 1.00        |

✅ Achieved **perfect scores** for both majority and minority classes.

---

## 🧪 Model Evaluation

To ensure generalization and avoid overfitting, the model was throrougly evaluated through:

- **5-fold cross-validation**  
- **10 repeated random train-test splits**  
- **Learning curve analysis**

All evaluation methods consistently confirmed the model’s **low variance**, **strong generalization**, and **stable performance** across multiple trials.

---

## 🔬 Benchmark Comparison

This model **outperformed all prior studies** using this dataset, including:

- **Gauhar et al. (2021)** – 92.00% using standard KNN  
- **Alam et al. (2021)** – 95.63% using hybrid random forest ensemble
- **Asif et al. (2023)** – 97.69% using random forest    
- **Rahman (2023)** – 98.86% using GRU (Gated Recurrent Unit)

By contrast, this approach semi-supervised learning approach achieved a **99.86% F1-score**.

---

## 🧠 Why Use KNN?

This research demonstrates that even a relatively **simple algorithm like KNN**, when paired with a thoughtful **semi-supervised learning framework**, **selective pseudo-labeling**, and **careful preprocessing**, can **outperform more complex models** like deep learning (GRU) and ensemble methods.

### ✅ Key Advantages:
- **Outperforms** complex models on this benchmark dataset
- Requires **less computational power**
- Involves **minimal hyperparameter tuning**
- Delivers **greater interpretability**
- Scales well in **data-scarce environments**

Given the near-perfect test F1-score of 0.9986 achieved by the semi-supervised KNN model with 1.00 precision, recall, and F1-score across both classes further experimentation with more complex models was not prioritized. The model's performance already surpassed previous benchmarks on this dataset, demonstrating that a carefully designed lightweight approach can be both effective and efficient.

---

## 🗂️ Project Structure

```bash
Flood_Prediction/
├── Flood_Prediction.ipynb       # Main Jupyter notebook
└── README.md                    # Project documentation (this file)
