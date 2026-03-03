# 📘 Lecture 3 – Introduction to ML Frameworks

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

# 📘 Lecture 6 – Introduction to Deep Learning

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

# 📘 Lecture 7 – Automatic Differentiation & Optimization

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

-------------------------------------------------------------

# 📘 Lecture 8 – Pipelines & Automated Training

In this lecture, I practiced structuring machine learning workflows using **pipelines** and automating experiments with configurable training functions.

The focus was on combining preprocessing, modeling, and evaluation into reusable and reproducible workflows.

## ✅ What I Learned

- Building pipelines with Scikit-Learn
- Combining preprocessing and models in one workflow
- Using `StandardScaler` inside pipelines
- Training and evaluating models through pipelines
- Creating reusable experiment functions
- Automating training with configuration dictionaries
- Running multiple experiments efficiently
- Saving model metrics in JSON format
- Comparing different model configurations

## 🧠 Key Concepts

- Machine Learning pipelines
- End-to-end workflows
- Reproducibility
- Experiment automation
- Modular ML design
- Configuration-driven training
- Separation of preprocessing and modeling

## 📊 Models

### Classification

- Logistic Regression
- Support Vector Machine (SVM)

Used for classification on the Iris dataset with standardized features.

## 📈 Evaluation

### Classification Metrics

- Accuracy
- F1-score (Macro Average)

### Experiment Tracking

- Saved metrics (`metrics_*.json`)
- Model parameters
- Performance comparison

## ⚙️ Technical Topics

- `Pipeline` from Scikit-Learn
- `StandardScaler`
- Custom experiment functions
- Config dictionaries
- JSON logging
- Model serialization (`joblib`)

## 🎯 Goal

Learn how to build scalable ML workflows:

**Preprocess → Pipeline → Train → Evaluate → Log → Compare**

This lecture prepares for:

- MLOps practices
- Experiment tracking
- Model reproducibility
- Production-style ML pipelines

# 📘 Lecture 9 – Simplified Processes (Lightning & fastai)

In this lecture, I practiced reducing boilerplate code in deep learning workflows by refactoring a standard PyTorch training loop into higher-level frameworks like PyTorch Lightning.

The focus was on moving away from manual loop management to improve code readability, reproducibility, and scalability.

## ✅ What I Learned

* **Building a baseline model:** Created a standard deep learning model and training loop in pure PyTorch.


* **Manual Device Management**: Handled device placement (`CPU` vs. `GPU`/`MPS`) manually for both the model and data.


* **Refactoring to Lightning:** Converted standard PyTorch code into an organized `LightningModule`.


* **Automated Training:** Used the Lightning `Trainer` to automate the training process and eliminate manual loops.


* **Code Organization**: Centralized model logic, loss functions, and optimizers into a single class.


* **Automated Backpropagation:** Handled batch iteration and gradients without writing manual `loss.backward()` or `optimizer.step()` calls.


* **Metric Logging:** Automated the logging of training metrics like loss and accuracy.


* **Readability Comparison**: Compared the boilerplate of "vanilla" PyTorch against the cleaner Lightning implementation.

## 🧠 Key Concepts
* **Boilerplate Reduction**: Eliminating repetitive code for training steps and hardware management.

* **LightningModule**: A structured way to organize PyTorch code into specific hooks like `training_step`.

* **The Trainer:** A high-level interface that handles training loops, validation, and hardware acceleration automatically.

* **State Management**: Automating hardware-specific calls like `model.train()`, `model.eval()`, and `optimizer.zero_grad()`.

* **Device Agnosticism:** Writing code that runs on any hardware (CUDA, MPS, CPU) without manual `.to(device)` calls.

## 📊 Models

### Multi-Layer Perceptron (MLP)

* **Architecture**: 3 Layers ($4$ input nodes $\to$ $128$ hidden nodes $\to$ $3$ output nodes).
* **Activation:** ReLU.
* **Optimization:** Adam Optimizer ($lr=0.01$).
* **Loss Function:** Cross-Entropy Loss for multi-class classification.

## 📈 Evaluation

### Baseline vs. Lightning
* **Dataset:** Iris Dataset (Standardized features).

* **Metrics**: Accuracy tracking across epochs.

* **Performance**: Achievement of approximately **86.67% accuracy** on the test set.

## ⚙️ Technical Topics
* `nn.Sequential` for model definition.

* `TensorDataset` & `DataLoader` for data handling.

* `pl.LightningModule & `pl.Trainer`.

* `training_step and `configure_optimizers`.

* **Hardware detection logic for `cuda`, `mps`, and `cpu`.

## 🎯 Goal
Learn how to transition from low-level manual loops to professional-grade frameworks:

**Standard PyTorch** → **Refactor to Lightning** → **Automate Loop** → **Scale Hardware**

This lecture prepares for:

* Large-scale Deep Learning projects.

* Professional MLOps workflows.

* Collaborative AI research with standardized code structures.