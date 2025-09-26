# 🩺 Chest X-Ray Classification (Normal vs Pneumonia)

This project applies **Machine Learning techniques** to classify chest X-ray images into **Normal** and **Pneumonia** categories.  

---

## 📂 Dataset

- **Name:** Chest X-Ray Images  
- **Total Images:** ~5,863  
- **Classes:**  
  - Normal  
  - Pneumonia  

---

## 📝 Tasks

### 🔹 Task 1: Data Acquisition and Exploration
- Download and load the dataset.  
- Display 5–10 random sample images from each class.  
- Report number of images per class (check class balance).  

### 🔹 Task 2: Data Preprocessing
- Resize images to a fixed dimension (e.g., **128×128**).  
- Convert to grayscale if required.  
- Apply **feature extraction methods**:
  - Histogram of Oriented Gradients (**HOG**)  
  - Local Binary Patterns (**LBP**)  
  - Gray-Level Co-occurrence Matrix (**GLCM**)  
- Prepare feature vectors for each image.  
- Split dataset into:
  - **Training:** 70%  
  - **Validation:** 15%  
  - **Testing:** 15%  

### 🔹 Task 3: Model Training and Evaluation
- Implement baseline models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
- Perform **hyperparameter tuning** using cross-validation.  
- Evaluate models using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - AUROC  
- Generate **confusion matrices** and analyze misclassifications.  
- Compare classifiers and select the **best performing model**.  

---

## 📊 Results

| Model              | Accuracy | Precision | Recall | F1-Score | AUROC |
|--------------------|----------|-----------|--------|----------|-------|
| Logistic Regression | **0.9511** | **0.9628** | **0.9704** | **0.9666** | **0.9826** |
| Decision Tree       | 0.9067  | 0.9307    | 0.9423 | 0.9364   | 0.8139 |
| Random Forest       | 0.9374  | 0.9550    | 0.9594 | 0.9572   | 0.9769 |

✅ **Best Performer:** Logistic Regression with **95.1% accuracy** and **AUROC of 0.983**.  

---

## 📌 Key Insights
- Logistic Regression surprisingly outperformed tree-based models.  
- Random Forest also performed strongly, but Logistic Regression showed slightly better generalization.  
- Decision Tree had lower AUROC, indicating weaker performance in distinguishing classes.  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, scikit-learn, OpenCV, scikit-image  

---

## 📖 References
- [Chest X-Ray Dataset (Kaggle)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
