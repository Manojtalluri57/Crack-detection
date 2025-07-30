# 🔍 Automated Crack Detection in Concrete Surfaces using Deep Learning

This project presents a deep learning-based approach for automated detection and segmentation of cracks in concrete structures using **MATLAB** and the **DeepLabv3+** architecture. The goal is to enable fast, accurate, and scalable crack identification from site images for civil engineering applications.

---

## 📌 Objective
To develop and evaluate a semantic segmentation model that identifies cracks in concrete surfaces and estimates crack severity using pixel-wise classification.

---

## 🧠 Model Details
- **Architecture**: DeepLabv3+  
- **Backbone**: ResNet-18  
- **Framework**: MATLAB Deep Learning Toolbox  
- **Training Epochs**: 60  
- **Learning Rate**: 0.0001 (constant)  
- **Hardware**: Single CPU

---

## 📊 Results Summary

### 🔢 Performance Metrics

| Metric            | Value     |
|-------------------|-----------|
| **Global Accuracy**     | 88.11%    |
| **Mean Accuracy**       | 76.42%    |
| **Mean IoU**            | 62.72%    |
| **Weighted IoU**        | 81.25%    |
| **Mean BF Score**       | 69.16%    |
| **Validation Accuracy** | **90.41%** |

---

### 📉 Class-wise Metrics

| Class       | Accuracy | IoU    | BF Score |
|-------------|----------|--------|----------|
| Background  | 61.01%   | 0.3828 | 0.6347   |
| Crack       | 91.83%   | 0.8716 | 0.7486   |

---

## 📊 Confusion Matrix

- **True Positives (Crack)**: 416,017  
- **False Positives**: 24,283 (Predicted crack where background existed)  
- **False Negatives**: 36,990 (Missed cracks)  
- **True Negatives**: 38,000

Model shows **high recall (91.8%)** for cracks with slightly lower performance on background classification due to surface texture and lighting noise.

---

## 📈 Training Progress

- **Converged in 60 epochs**
- **Smoothed training accuracy** remained near 100%
- Validation accuracy gradually increased to **90.41%**
- Loss decreased steadily across training and validation

---

## 🔬 Post-processing

- Crack width estimation (in mm) from segmented masks  
- Classification into severity levels (minor/moderate/severe) based on width  
- Potential for integration into field-inspection tools

---

## 📁 Repository Contents

- `trainModel.m`: Training script using DeepLabv3+  
- `generateMasks.m`: Semi-automated mask generation from grayscale images  
- `evaluateModel.m`: Evaluation script with confusion matrix and metric plots  
- `images/`: Sample crack and background images  
- `masks/`: Binary ground truth masks  
- `results/`: Output predictions and metrics  

---

## 🚀 Future Work

- Integrate model into mobile inspection app  
- Extend to spalling, corrosion, and surface defects  
- Optimize using GPU and parallel training  
- Cross-validate on multiple real-world datasets

---

## 🧑‍💻 Author

**Talluri Manoj Kartheek Chowdary**  
M.Tech Structural Engineering, IIT Kharagpur  
[LinkedIn](https://www.linkedin.com/in/talluri-manoj-18b216196) | [GitHub](https://github.com/Manojtalluri57)

---

> 📌 _For use in infrastructure inspection, maintenance scheduling, and smart civil engineering applications._
