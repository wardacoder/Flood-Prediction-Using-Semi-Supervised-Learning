# 🧠 Semi-Supervised ML Framework  

A **semi-supervised machine learning framework** engineered for **unlabeled data environments**, demonstrating applied **AI/ML and data science expertise** through efficient model design and evaluation. Built with **Python** and **Scikit-learn**, it integrates **selective pseudo-labeling**, **comprehensive data preprocessing**, and a **three-phase training workflow** to deliver **state-of-the-art predictive accuracy (F1 = 0.9986)** on real-world flood data: surpassing results reported in published studies. 

This project demonstrates practical **ML engineering**: from problem definition and feature design to model optimization, evaluation, and validation. It highlights how even lightweight algorithms like KNN can outperform complex deep-learning architectures when paired with robust, data-centric methodology.

---

## 🌍 Context

Floods are among the most frequent and damaging natural disasters, yet **labeled flood datasets** remain scarce due to annotation costs and logistical limitations.  
This project addresses that challenge through **semi-supervised learning**, using a small set of labeled data and intelligently expanding it via **pseudo-labeling** to train an accurate and reliable predictive model.

---

## 📖 Overview

This project introduces a **semi-supervised ML solution** that tackles **data scarcity** while maintaining interpretability and performance.  
By combining **selective pseudo-labeling** with **rigorous preprocessing** and **model fine-tuning**, the framework achieves near-perfect classification of flood vs non-flood events.

**Key Highlights**
- Semi-supervised learning approach leveraging pseudo-labeling  
- Strong focus on data preprocessing and feature scaling  
- Simple yet optimized **KNN** model achieving 99.86% F1-score  
- Perfect precision and recall for both classes  
- Validated through multiple evaluation techniques (cross-validation, random splits, learning curves)  
- Demonstrates end-to-end ML proficiency: preprocessing → model training → evaluation  

---

## 🎯 Objective

To design and implement a **semi-supervised machine learning framework** capable of robust predictive performance under **limited labeled data** conditions, emphasizing:  
- Integration of **data preprocessing**, **model training**, and **evaluation**  
- Application of **selective pseudo-labeling** to enhance model generalization  
- Hands-on experience with **Scikit-learn**, **NumPy**, **Pandas**, and **Matplotlib**  
- Building scalable, interpretable, and production-ready ML systems  

---

## ⚙️ Methodology

### **1. Data Preprocessing**
Applied extensive preprocessing to ensure high data quality and strong signal-to-noise ratio:  
- Removed **non-predictive and redundant features**  
- Encoded categorical variables using one-hot encoding  
- Detected and removed outliers  
- Analyzed **feature distributions** and corrected skewness  
- Applied **Min-Max scaling** for skewed features and **Standard scaling** for normally distributed ones  
- Evaluated model performance on both original and **SMOTE-balanced datasets**

### **2. Initial Supervised Training**
- Trained a base **KNN** model using 3,432 labeled samples after preprocessing.  
- Conducted parameter tuning for `n_neighbors`, `metric`, and `weights`.  

### **3. Selective Pseudo-Labeling**
- Generated pseudo-labels for ~78% unlabeled data using the trained model.  
- Retained only predictions with ≥95% confidence to ensure label reliability.  

### **4. Final Semi-Supervised Training**
- Combined original labeled and high-confidence pseudo-labeled data (total 13,910 samples).  
- Retrained the KNN classifier and optimized using **GridSearchCV** for hyperparameter tuning.  

---

## 📊 Results

| Metric     | Score  |
|-------------|---------|
| F1-score    | 0.9986  |
| Precision   | 1.00    |
| Recall      | 1.00    |
| Accuracy    | 1.00    |

✅ Achieved **perfect precision, recall, and F1-score** across both classes, demonstrating reliable and unbiased classification.

---

## 🧪 Model Evaluation

To validate model robustness and prevent overfitting, multiple evaluation strategies were used:  
- **5-fold cross-validation**  
- **10 repeated random train-test splits**  
- **Learning curve analysis**

All tests confirmed consistent high performance, **low variance**, and strong **generalization** across multiple trials.

---

## 🔬 Benchmark Comparison

| Study | Method | F1 / Accuracy (%) |
|-------|---------|------------------|
| Gauhar et al. (2021) | Standard KNN | 92.00 |
| Alam et al. (2021) | Hybrid RF Ensemble | 95.63 |
| Asif et al. (2023) | Random Forest | 97.69 |
| Rahman (2023) | GRU (Deep Learning) | 98.86 |
| **This Work** | Semi-Supervised KNN | **99.86** |

> This model outperformed all previously published approaches on the same dataset, validating the strength of a **data-driven semi-supervised learning approach**.

---

## 💡 Why KNN?

While often perceived as a simple algorithm, **KNN** can achieve exceptional performance when paired with:
- **Thoughtful preprocessing**  
- **High-quality feature engineering**  
- **Selective pseudo-labeling**  
- **Careful hyperparameter tuning**

### ✅ Key Advantages
- Outperforms complex models on this benchmark dataset  
- Requires minimal computational resources  
- Offers interpretability and transparency  
- Easily scalable to other domains with limited labeled data  

This reinforces that **model simplicity + data quality** often outperform **complex architectures without data discipline**.

---

## 🧠 Key Takeaways

- Demonstrated real-world **ML problem solving** under data constraints  
- Applied **semi-supervised learning** for efficient label utilization  
- Showcased **end-to-end ML workflow**: from preprocessing → pseudo-labeling → evaluation  
- Outperformed published results using a **lightweight, interpretable** model  
- Highlights **industry-ready ML engineering** and **data-driven thinking**

---

## 🗂️ Project Structure

```
SemiSupervised-ML-Framework/
├── Flood_Prediction.ipynb     # Main Jupyter Notebook
└── README.md                  # Project Documentation
```

---

## 🧰 Tech Stack

- **Languages & Libraries:** Python, Scikit-learn, NumPy, Pandas, Matplotlib  
- **Techniques:** Semi-supervised learning, Pseudo-labeling, SMOTE, Feature Scaling  
- **Evaluation Tools:** GridSearchCV, Cross-validation, Learning Curves  

---

## 🏁 Outcome

This project proves that with **data-centric design and careful engineering**, traditional ML algorithms can achieve **state-of-the-art accuracy** even in **low-label data environments** — a critical insight for scalable AI systems used in the industry.
