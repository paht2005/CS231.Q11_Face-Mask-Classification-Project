<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)" width="400">
  </a>
</p>

<h1 align="center"><b>CS231.Q11 – Introduction to Computer Vision</b></h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
</p>

---
# **CS231 Course Project: Face Mask Classification** – A Comparative Study of Traditional ML and CNNs

> This repository contains the implementation of a **Face Mask Classification System**, developed as the final project for **CS231.Q11 – Introduction to Computer Vision** at the **University of Information Technology (UIT – VNU-HCM)**.
>
> The project performs a comparative study between traditional Machine Learning approaches (using **HOG** and **LBP** feature descriptors with **SVM, KNN, and Random Forest** classifiers) and modern **Convolutional Neural Networks (CNN)** to identify individuals wearing masks versus those without masks.
>
> The primary objective of this project is to **analyze and compare the effectiveness of traditional Machine Learning pipelines versus Deep Learning approaches** for the task of **face mask detection**, a binary image classification problem with real-world relevance in public health and surveillance systems.
>
> The project emphasizes **methodological comparison**, **feature representation**, and **performance evaluation**, rather than solely maximizing accuracy through deep models.


<p align="center">
  <img src="static/images/thumbnail.png" alt="thumbnail" width="600">
</p>

---

## Team Information
| No. | Student ID | Full Name | Role | Github | Email |
|----:|:----------:|-----------|------|--------|-------|
| 1 | 23521143 | Nguyen Cong Phat | Leader | [paht2005](https://github.com/paht2005) | 23521143@gm.uit.edu.vn |
| 2 | 23521168 | Nguyen Le Phong | Member | [kllp031](https://github.com/kllp031) | 23521168@gm.uit.edu.vn  |
| 3 | 23520213 | Vu Viet Cuong | Member | [Kun05-AI](https://github.com/Kun05-AI) |  23520213@gm.uit.edu.vn  | 


---

## **Table of Contents**
- [Repository Structure](#repository-structure)
- [Problem Statement](#problem-statement)
- [System Overview](#system-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Architectures](#model-architectures)
- [Training & Optimization](#training--optimization)
- [Installation](#installation)
- [Usage](#usage)
- [Demo Application](#demo-application)
- [Experimental Results](#experimental-results)
- [Discussion](#discussion)
- [Conclusion & Future Work](#conclusion--future-work)
- [License](#license)

---

## **Repository Structure**
```text
CS231.Q11_Face-Mask-Classification-Project/
├── src/             # Model training notebooks (Jupyter)
│   ├── CNN/                 	# Deep Learning CNN (Grayscale) training source code
│   ├── HOG_KNN/               # KNN training with HOG features source code
│   ├── HOG_RF/              	# Random Forest training with HOG features source code
│   ├── HOG_SVM/  				# SVM training with HOG features source code
│   ├── LBP_KNN/  				# KNN training with LBP features source code
│   ├── LBP_RF/  				# Random Forest training with LBP features source code
│   └── LBP_SVM/				# SVM training with LBP features source code
│
├── models/
│   ├── yunet.onnx              # Pre-trained Face Detection model (Included)
│   ├── mask_detector_model.h5  # Trained Mask Classification model (Included)
│   └── [others].joblib/.keras        # Large models/caches (Ignored - Download link below)
│
├── docs/                       # Report & presentation
│   ├── 23520213-23521143-23521168_Report.pdf
│   └── 23520213-23521143-23521168_Slide.pdf
│
├── static/                     # Static Assets
│   ├── images/                 # Images for Slide, Report, and Thumbnails
│   ├── results/                # Output images from Flask Web Demo
│   ├──  latex/                # Latex files
│   ├── templates/              # Web UI (index.html, indexSVM.html)
│   └── test/                   # Sample test images (e.g., test.jpg)
│
├── uploads/                    # Temporary storage for user-uploaded images
├── demo_webcam.py              # Real-time Webcam detection script
├── demoSVM_image_flask.py      # Flask Web Application script
├── requirements.txt            # Python dependencies
├── LICENSE
├──  .gitignore                  # Git ignore rules
└── README.md # Main project documentation
```

---

## **Problem Statement**

Face mask detection is a practical computer vision problem that requires **robust facial feature representation** under variations in:
- Illumination
- Pose
- Occlusion
- Mask styles and colors

The goal of this project is to:
1. Evaluate whether **hand-crafted features (HOG, LBP)** combined with classical classifiers can compete with CNN-based approaches.
2. Analyze trade-offs between **accuracy, computational cost, and deployment complexity**.
3. Develop a system capable of **real-time inference** using standard consumer hardware.

---

## **System Overview**

The proposed system consists of three main components:

1. **Offline Training Pipeline**
   - Image preprocessing
   - Feature extraction
   - Model training and hyperparameter optimization

2. **Inference Pipeline**
   - Face detection using **YuNet**
   - Feature extraction / CNN inference
   - Classification and post-processing

3. **Deployment Interfaces**
   - Flask-based web application (static image classification)
   - Real-time webcam detection

---

## **Key Features**

- **Binary Face Mask Classification** with high accuracy
- **Comparative Study** between:
  - Traditional ML: HOG/LBP + SVM, KNN, Random Forest
  - Deep Learning: Custom CNN
- **Automated Hyperparameter Tuning**
  - Optuna for ML models
  - Keras Tuner (Hyperband) for CNN
- **Real-time Detection** using webcam input
- **User-friendly Web Interface** built with Flask

---

## **Dataset**

### Face Mask 12K Images Dataset

- **Source**: Kaggle  
  🔗 https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset
- **Total Images**: Approximately **12,000 RGB images**
- **Image Characteristics**:
  - Diverse facial orientations
  - Multiple ethnicities
  - Various mask types and lighting conditions

### Dataset Structure
- **Training Set**: 10,000 images  
- **Validation Set**: 800 images  
- **Test Set**: 992 images  

> The dataset is well-balanced between the two classes, making it suitable for unbiased binary classification evaluation.
> No identity or personal information is associated with the dataset, ensuring ethical use for academic research.

---

## **Data Preprocessing**

To ensure consistency and reduce computational complexity, the following preprocessing steps were applied:

1. **Resizing**
   - All images resized to **128 × 128 pixels**

2. **Normalization**
   - Pixel intensities scaled to the range **[0, 1]**

3. **Grayscale Conversion**
   - Applied for traditional ML pipelines
   - Reduces dimensionality while preserving structural facial features

---

## **Feature Extraction**

### Histogram of Oriented Gradients (HOG)
- Captures **edge and shape information**
- Effective for representing facial geometry
- Tested configurations:
  - `6 × 3` cells
  - `8 × 2` cells (best-performing)

### Local Binary Patterns (LBP)
- Encodes **local texture patterns**
- Robust to illumination changes
- Useful for modeling fine-grained facial textures

---

## **Model Architectures**

### Traditional Machine Learning Models
- **Support Vector Machine (SVM)** with RBF kernel
- **K-Nearest Neighbors (KNN)**
- **Random Forest**

These models operate on extracted HOG or LBP feature vectors.

### Deep Learning Model
- Custom **Convolutional Neural Network (CNN)**
- Lightweight architecture optimized for grayscale input
- Designed to balance performance and training efficiency

---

## **Training & Optimization**

- **Traditional ML Models**
  - Hyperparameters optimized using **Optuna**
  - Objective: maximize validation accuracy

- **CNN**
  - Optimized using **Keras Tuner (Hyperband)**
  - Tuned parameters include:
    - Number of convolutional layers
    - Filter sizes
    - Learning rate
    - Dropout rate

---

## **Installation**

### 1. Clone repository
```bash
git clone https://github.com/paht2005/CS231.Q11_Face-Mask-Classification-Project.git
cd CS231.Q11_Face-Mask-Classification-Project
```

### 2. Create virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
--- 

## Usage
### 1. Train models
Open and run notebooks in **src/**:
```bash
jupyter notebook
```
- CNN/train-model-CNN-best-grayscale.ipynb
- HOG_KNN/train-model-HOG-KNN_6x3.ipynb
- HOG_KNN/train-model-HOG-KNN_8x2.ipynb
- HOG_RF/train-model-HOG-RF_8x2.ipynb
- HOG_RF/train-model-HOG-RF_6x3.ipynb
- HOG_SVM/train-model-HOG-SVM_6x3.ipynb
- HOG_SVM/train-model-HOG-SVM-8x2.ipynb
- LBP_KNN/train-model-LBP-KNN.ipynb
- LBP_RF/train-model-LBP-RF.ipynb
- LBP_SVM/train-model-LBP-SVM.ipynb

### 2. Run Flask demo
```bash
python demoSVM_image_flask.py
```
Open browser at:
```bash
http://127.0.0.1:5000
```
### 3. Real-time Webcam Detection
```bash
python demo_webcam.py
```

---

## **Demo Application**

The project provides two distinct interfaces to demonstrate the classification capabilities, catering to both static analysis and real-time monitoring.

### **1. Flask Web Application (Static Image Classification)**
A user-friendly web interface built on the **Flask Framework**, allowing users to upload individual images for detailed analysis.
- **Logic**: Receives image files via `indexSVM.html`, extracts **HOG 8x2** features using `skimage`, and performs inference using the optimized **SVM** model (`.joblib`).
- **Output**: Generates a bounding box and label directly on the browser, displaying the prediction result along with a confidence score.

<p align="center">
  <img src="static/images/demo/flask_web_demo.png" alt="Flask Web Demo Interface" width="800">
</p>

### **2. Real-time Inference Pipeline (Webcam)**
Designed for high-speed monitoring, this pipeline utilizes a specialized deep learning flow to ensure stability and performance in live video streams.
- **Face Detection**: Integrates **YuNet** (`yunet.onnx`) via OpenCV's `FaceDetectorYN` for ultra-lightweight and fast facial localization.
- **Classification**: Uses the **CNN** model (`mask_detector_model.h5`) on grayscale input. To optimize performance, detected faces are processed in **batches**.
- **Stabilization Techniques**:
    - **Temporal Smoothing**: Employs a `deque` buffer to average predictions over recent frames, effectively eliminating "label flickering".
    - **Centroid-based Tracking**: Maintains consistent object identity across the temporal domain using Euclidean distance tracking.
- **Performance**: Achieves a smooth processing rate of **over 25 FPS**, meeting the requirements for real-time surveillance.

<p align="center">
  <img src="static/images/demo_gif.gif" alt="Video real-time demo" width="800">
</p>

---
## **Experimental Results**

### **Test Accuracy Comparison**

| Model            | Feature Descriptor | Accuracy |
|------------------|-------------------|----------|
| CNN              | Automatic (None)  | 0.9869 |
| **SVM**          | **HOG (8×2)**     | **0.9899** |
| SVM              | HOG (6×3)         | 0.9879 |
| SVM              | LBP               | 0.9720 |
| KNN              | HOG (8×2)         | 0.9839 |
| KNN              | HOG (6×3)         | 0.9748 |
| KNN              | LBP               | 0.9234 |
| Random Forest    | HOG (8×2)         | 0.9819 |
| Random Forest    | HOG (6×3)         | 0.9819 |
| Random Forest    | LBP               | 0.9093 |

Overall, HOG-based methods consistently outperform LBP-based methods, with SVM emerging as the most effective classifier.


---

### **Experimental Analysis**

Based on the experimental results summarized above, several key observations can be drawn:

1. **Superior Performance of HOG + SVM**

   The combination of the **HOG (8×2)** feature descriptor and the **Support Vector Machine (SVM)** classifier achieves the highest classification accuracy (**0.9899**).  
   This result demonstrates that, for datasets with relatively stable facial structures, **well-designed hand-crafted shape features** can provide highly discriminative representations.  
   In this setting, explicit gradient-based edge information enables more effective class separation than a baseline CNN trained from scratch.

2. **Robustness and Stability of the CNN Model**

   The CNN model achieves a strong performance with an accuracy of **0.9869**, indicating excellent generalization ability.  
   A major advantage of CNNs lies in their **end-to-end learning capability**, which eliminates the need for manual feature engineering and facilitates scalability when larger or more diverse datasets become available.

3. **Effectiveness of the HOG (8×2) Configuration**

   Across all traditional machine learning classifiers (SVM, KNN, and Random Forest), the **HOG (8×2)** configuration consistently outperforms or matches the **HOG (6×3)** configuration.  
   The vertical cell partitioning of **8×2** is particularly effective in capturing **vertical symmetry and structural patterns** of faces and masks, which are crucial cues for mask detection.

4. **Limitations of LBP Features**

   The **LBP (Local Binary Pattern)** descriptor yields the lowest accuracy across most classifiers, especially when combined with KNN and Random Forest.  
   This suggests that **edge and shape information (gradients)** is more informative than **surface texture information** for the face mask classification task.

---

### **Overall Conclusion**

The experimental evaluation confirms that, for the current dataset, the optimized traditional pipeline **HOG + SVM** achieves the highest absolute accuracy (**0.9899**), slightly outperforming the grayscale CNN model (**0.9869**).

---

### **In-depth Discussion**

- **Effectiveness of Shape-based Features**  
  HOG descriptors rely on gradient orientation distributions, which are particularly suitable for representing **structured and symmetric objects** such as human faces and face masks.  
  With a moderately sized dataset, HOG provides a highly separable feature space without requiring complex learning processes.  
  It should be noted that the CNN model in this project was trained on grayscale images using a **moderate architecture**, without leveraging pre-trained backbones or advanced data augmentation techniques.

- **Dataset Size Constraints for CNNs**  
  While CNNs are powerful representation learners, their full potential typically emerges when trained on **large-scale datasets**.  
  With approximately **10,000 training images**, the CNN may have reached convergence but lacked sufficient data diversity to learn features more discriminative than the optimized HOG representation.

- **Practical Implications**  
  These findings indicate that in scenarios with **limited data and computational resources**, the combination of **hand-crafted features (HOG)** and a **strong classifier (SVM)** remains a highly effective and practical solution, offering both high accuracy and efficient inference.



---

## **Discussion**

Experimental results indicate that:

- The **HOG + SVM** pipeline provides the best overall performance.
- **Traditional machine learning approaches** remain highly competitive when paired with effective feature engineering.
- CNN performance is strong but **sensitive to architectural design and data volume**, particularly when trained from scratch.

These findings highlight that **carefully engineered hand-crafted features can outperform deep learning models** in structured vision tasks with limited or moderately sized datasets.

---

## **Conclusion & Future Work**
In conclusion, this project demonstrates that classical computer vision pipelines, when carefully engineered and optimized, can rival or even surpass deep learning models in structured vision tasks with limited data.


This project demonstrates that **classical computer vision techniques**, when properly optimized, remain highly effective for real-world applications such as face mask detection.

Potential future extensions include:
- Applying **Transfer Learning** with advanced CNN backbones (e.g., ResNet, MobileNet)
- Extending the system to **multi-class mask type classification**
- Optimizing deployment for **edge and embedded devices**

---

## **License**

This project is developed for **academic purposes** under the course  
**CS231.Q11 – Introduction to Computer Vision** at the **University of Information Technology (UIT)**.

Released under the **MIT License**.
See the [LICENSE](./LICENSE) file for details.
