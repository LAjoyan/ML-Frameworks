📘 Lecture 3 – Introduction to ML Frameworks

In this lecture, I explored the basic mathematical and computational foundations behind modern machine learning frameworks, focusing on NumPy, PyTorch, and Scikit-Learn.

✅ What I Learned

- Creating and manipulating vectors and matrices using NumPy
- Performing dot products and matrix multiplication
- Computing cosine similarity
- Understanding L2 normalization
- Writing and testing simple mathematical functions
- Using PyTorch in eager execution mode
- Comparing eager vs compiled graph execution
- Loading datasets with Scikit-Learn
- Training a Logistic Regression model
- Understanding convergence warnings and model performance

🧠 Key Concepts

- Linear algebra for machine learning
- Vector similarity measures
- Numerical computation with tensors
- Framework execution modes
- Supervised learning fundamentals
- Model evaluation basics

📊 Models

Logistic Regression (Scikit-Learn)

Used to perform basic classification on the Iris dataset.

⚙️ Technical Topics

- NumPy arrays and operations
- PyTorch tensors and performance testing
- Scikit-Learn dataset handling
- Model training and evaluation

🎯 Goal

Build strong foundations in:

Math → Arrays → Tensors → Models → Evaluation

This lecture prepares the groundwork for advanced topics such as:

- Classical Machine Learning
- Unsupervised Learning
- Deep Learning

-------------------------------------------------------------

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

-------------------------------------------------------------

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

-------------------------------------------------------------

📘 Lecture 7 – Automatic Differentiation & Optimization

In this lecture, I explored how modern deep learning frameworks compute gradients automatically and how different optimizers affect model training.

I focused on understanding automatic differentiation in PyTorch and comparing optimization algorithms on a synthetic classification dataset.

✅ What I Learned

- Using PyTorch Autograd to compute derivatives
- Understanding computational graphs
- Comparing automatic vs analytical gradients
- Creating synthetic datasets with Scikit-Learn
- Preventing data leakage with proper preprocessing
- Scaling data after train/test split
- Building custom training loops
- Training neural networks with different optimizers
- Evaluating model performance

🧠 Key Concepts

- Automatic differentiation (Autograd)
- Gradient computation
- Backpropagation mechanics
- Optimization algorithms
- Data leakage in ML pipelines
- Reproducible ML experiments

📊 Models

Neural Network (Binary Classifier)

Used to classify synthetic data into two classes.

- Input: Feature vectors
- Hidden layers with activation functions
- Output: Binary classification
- Loss: Binary Cross Entropy

⚙️ Technical Topics

- PyTorch tensors with gradients
- requires_grad and backward()
- Optimizers (SGD, Adam, etc.)
- Training and evaluation loops
- Scikit-Learn dataset generation
- StandardScaler usage

📈 Evaluation

- Accuracy (classification performance)
- Loss (training and validation error)
- Optimizer comparison
- Generalization performance

🎯 Goal

Learn how models actually learn by:

Gradients → Optimization → Parameter Updates → Convergence → Performance

This lecture builds the foundation for:

- Advanced Deep Learning
- Model Tuning
- Training Optimization
- Research-Level ML Experiments