# 📘 Lecture 4 – Scikit-Learn API (Part 1)

In this lecture, I practiced working with machine learning using Scikit-Learn.

## ✅ What I Learned

- Loading ready datasets (Iris, Diabetes)
- Splitting data with `train_test_split` and `random_state`
- Exploring data (EDA) with matplotlib and seaborn
- Using pairplot and scatter plots
- Training models with `fit()` and `predict()`
- Comparing models using loops

## 📊 Models

### Classification
- Logistic Regression
- SVC

### Regression
- Linear Regression
- Ridge

## 📈 Evaluation

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Regression Metrics
- MAE
- R²

## 🎯 Goal

Learn the basic ML workflow:

**Load → Explore → Train → Predict → Evaluate → Compare**

-------------------------------------------------------------

# 📘 Lecture 5 – Scikit-Learn API (Part 2)

In this lecture, I practiced unsupervised learning using Scikit-Learn.
## ✅ What I Learned

- Understanding K-Means clustering  
- Finding the optimal number of clusters using the Elbow method  
- Visualizing clusters with matplotlib  
- Applying PCA (Principal Component Analysis)  
- Reducing dimensions from 4D to 2D and 3D  
- Comparing K-Means clusters with true labels  

## 📊 Models & Techniques

### 🔹 Clustering
- K-Means  
- Elbow Method (WCSS / Inertia)  

### 🔹 Dimensionality Reduction
- PCA (2D visualization)  
- PCA (3D visualization)  

## 📈 Visualization

- Pairplots for EDA  
- Elbow curve plot  
- 2D PCA scatter plots  
- 3D PCA projection  

## 🎯 Goal

Learn the unsupervised ML workflow:

**Load → Explore → Scale → Cluster → Reduce Dimensions → Visualize → Compare**

📘 Lecture 6 – Introduction to Deep Learning

In this lecture, I practiced building and training a simple neural network using PyTorch and the MNIST dataset.

✅ What I Learned

- Working with tensors in PyTorch
- Loading and preprocessing the MNIST dataset
- Creating a custom neural network
- Using forward propagation
- Training with backpropagation and optimizers
- Using loss functions for classification
- Running training loops with epochs
- Evaluating model performance
- Visualizing predictions and errors

📊 Model

Neural Network (Fully Connected)

- Input: 28 × 28 images (flattened)
- Hidden layers with ReLU activation
- Output: 10 classes (digits 0–9)
- Softmax classification

📈 Evaluation

Classification Metrics

- Accuracy
- Loss (training and validation)
- Confusion Matrix

Visual Analysis

- Sample predictions (correct and wrong)
- Training and validation loss curves

🎯 Goal

Learn the basic deep learning workflow:

Load → Preprocess → Build → Train → Predict → Evaluate → Visualize
