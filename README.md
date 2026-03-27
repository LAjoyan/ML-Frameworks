**You can find instructions about dvc** [here](./Code_alongs/Lecture-07/dvc_quick_guide.md)

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

In this lecture, I practiced building and training a simple neural network using PyTorch and the MNIST dataset. I also created dedicated reference guides to master tensor manipulation and cross-framework syntax.

The focus was on understanding the fundamental building blocks of deep learning, from raw tensors to fully trained classification models.

## 📂 Reference Guides Added
* **[Tensors 101](./Tensors%20101.md):** A practical deep dive into tensor creation, reshaping, broadcasting, and the mechanics of **Autograd**.
* **[PyTorch vs. Keras Syntax](./Pytorch_vs_Keras_Syntax.md):** A side-by-side comparison of model definitions, training loops, and evaluation patterns between PyTorch and Keras/TensorFlow.

## ✅ What I Learned

* **Tensor Foundations:** Mastered `shapes`, `dtypes`, and `broadcasting` logic in PyTorch.


* **Working with Tensors:** Practiced creating tensors from random generators, Python lists, and NumPy arrays.


* **MNIST Workflow:** Loading and preprocessing the MNIST dataset (28x28 grayscale images) for digit classification.


* **Custom Architectures:** Building a multi-layer neural network using `nn.Module` and `nn.Sequential`.


* **The Training Loop:** Implementing manual backpropagation using `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.


* **Autograd Mechanics:** Understanding how `requires_grad=True` allows PyTorch to automatically track and calculate gradients.


* **Evaluation Techniques:** Using `model.eval()` and `torch.no_grad()` to freeze weights during the testing phase.


* **Visualization:** Plotting training curves and confusion matrices to analyze where the model makes mistakes.

## 🧠 Key Concepts
* **Flattening:** Converting 2D image data ($28 \times 28$) into a 1D vector ($784$) for input into a Fully Connected layer.

* **Activation Functions:** Using ReLU to introduce non-linearity and Softmax for final probability distributions.

* **Loss Functions:** Applying Cross-Entropy Loss for multi-class classification problems.

* **Gradient Descent:** Using optimizers like SGD or Adam to update weights based on calculated errors.

## 📊 Models

### Fully Connected Neural Network (MLP)

* **Architecture**: Input Layer ($784$) $\to$ Hidden Layers (ReLU) $\to$ Output Layer ($10$ classes).
* **Activation:** ReLU for hidden layers, Softmax/Log-Softmax for output.
* **Optimization:** Adam or SGD Optimizer.
* **Loss Function:** Cross-Entropy Loss.

## 📈 Evaluation

### Classification Metrics
* **Dataset:** MNIST Handwritten Digits.

* **Performance**: Tracking accuracy and loss across multiple epochs.

* **Visuals**: Confusion matrices and sample predictions (Correct vs. Incorrect).

## ⚙️ Technical Topics
* `torch.tensor` & `torch.autograd`.

* `torchvision.datasets.MNIST`.

* `nn.Linear` & `nn.ReLU`.

* `torch.optim` (Adam/SGD).

* **Hardware detection logic for `cuda`, `mps`, and `cpu`.

## 🎯 Goal
Master the fundamental deep learning workflow and establish a "syntax bridge" between major frameworks:

**Tensors** → **Preprocess** → **Build** → **Train** → **Evaluate** → **Cross-Framework Syntax**

This lecture prepares for:

* Building custom architectures for computer vision.

* Understanding the "under the hood" mechanics of backpropagation.

* Transitioning between different deep learning libraries like Keras and PyTorch.

## 1. Tensors 101
A practical introduction to tensors, the fundamental building blocks of deep learning.

### Goals
* Create tensors from Python data and random generators.
* Inspect shapes and data types (`dtypes`).
* Perform basic operations and understand **broadcasting**.
* Understand how gradients attach to tensors for backpropagation.

### Core Operations
```python
import torch
torch.manual_seed(0)

# Creation
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros((2, 3))
c = torch.randn((2, 3))

# Shapes, Indexing, and Reshaping
x = torch.arange(12).reshape(3, 4)
# Example: x[0] (first row), x[:, 1] (second column)

# Operations and Broadcasting
v = torch.tensor([1.0, 2.0, 3.0])
m = torch.ones((2, 3))
result = m + v # v is broadcasted to match m's shape
```

### Autograd Basics

Tensors can track gradients when `requires_grad=True`, enabling automatic differentiation.

```python
w = torch.tensor([2.0, -1.0], requires_grad=True)
y = (w ** 2).sum()
y.backward()
print(w.grad) # Access the computed gradients
```

## 2. PyTorch vs. Keras Syntax Guide

A quick reference for switching between the two most popular deep learning libraries.

**Tensor Creation**

| Feature   | PyTorch                | Keras / TensorFlow      |
|-----------|-----------------------|------------------------|
| Constant  | `torch.tensor([[1, 2]])` | `tf.constant([[1, 2]])` |
| Zeros     | `torch.zeros((2, 3))`    | `tf.zeros((2, 3))`      |

### Model Definition

## PyTorch (Class-based)

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)
```

### Keras (Sequential API)

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(10,)),
    keras.layers.Dense(1),
])
```
## Training & Evaluation

| Action | PyTorch (Manual Loop) | Keras (High-level) |
|--------|----------------------|-------------------|
| Setup  | Define Optimizer & Loss function | `model.compile(opt, loss)` |
| Train  | `loss.backward() + optimizer.step()` | `model.fit(dataset, epochs=5)` |
| Eval   | `model.eval() + torch.no_grad()` | `model.evaluate(x_test, y_test)` |

### Saving Models
* PyTorch: `torch.save(model.state_dict(), "model.pt")`

* Keras: `model.save("model.keras")`
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

# 🔍 Deep Dive: Chain Rule & Optimization

---

## 1. Chain Rule Walkthrough (Code vs. Math)

I compared manual calculus with PyTorch's autograd to verify how gradients flow through a computational graph.

### Example Graph

[u = x \cdot y]  
[v = u + x]  
[z = v^2]  

### Python Implementation

```python
import torch

# Define variables with gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward pass
u = x * y
v = u + x
z = v ** 2

# Backward pass
z.backward()

print(f"PyTorch Gradients: dz/dx = {x.grad}, dz/dy = {y.grad}")
``` 

## 2. Optimizer Path Visualization

I implemented a visualization tool to track how different algorithms navigate a non-convex loss surface with multiple valleys.

### Optimizers Compared

- **SGD:** Pure gradient descent; can be slow or get stuck in local minima.  
- **Momentum:** Adds a "velocity" component to accelerate through flat regions and dampen oscillations.  
- **Adam:** Combines adaptive learning rates with momentum for robust convergence.  

### Visualization Concept

- Plot the 2D loss surface: \(L(x, y)\)  
- Show paths taken by different optimizers from the same starting point  
- Analyze convergence speed and stability

---

## 3. Updated Technical Topics

- **Computational Graphs:** Track dependencies between variables to automate backpropagation.  
- **Manual vs. Auto Grad:** Validate \(\frac{\partial z}{\partial x}\) and \(\frac{\partial z}{\partial y}\) using the chain rule.  
- **Loss Surfaces:** Visualize 2D non-convex functions to test optimizer stability.

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

**You can find MLOps Lifecycle Guide** [here](./Code_alongs/Lecture-08/MLOps_Lifecycle.md)

**You can view the Pipeline Components Diagram** [here](./Code_alongs/Lecture-08/pipeline_components_diagram.md)

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

* Professional MLOps workflows.'

* Collaborative AI research with standardized code structures.

# 📘 Lecture 10 – Optimization and Fine-Tuning

In this lecture, I practiced advanced deep learning workflows focusing on **Transfer Learning** and **Fine-Tuning**.

The goal was to take a pre-trained **ResNet18** model, adapt it for **CIFAR-10 classification**, and optimize it for production using **ONNX**.

The focus was on moving from manual model building to leveraging industry-standard architectures and cross-platform export formats.

---

# ✅ What I Learned

### Building a Baseline Model
- Loaded a pre-trained `resnet18` model using `torchvision.models` with ImageNet weights.

### Manual Device Management
- Handled device placement manually (CPU vs. GPU/MPS) for both the model and the data.

### Transfer Learning
- Replaced the final fully connected (FC) layer of ResNet18 to match the 10 classes of the CIFAR-10 dataset.

### Layer Freezing
- Implemented manual freezing of base layers by setting:
  requires_grad = False
  to preserve pre-trained features.

### Selective Fine-Tuning
- Unfroze specific parts of the base model (specifically `layer4`) to adapt deeper features to the new dataset.

### Optimizer Filtering
- Configured the Adam optimizer to update only parameters where:
  requires_grad = True

### Model Exportation
- Successfully exported the trained model to:
  - `.pt` (PyTorch `state_dict`)
  - `.onnx` (for deployment)

### Performance Analysis
- Compared the accuracy of a frozen-base model against a fine-tuned version to measure optimization gains.

---

# 🧠 Key Concepts

### Boilerplate Reduction
Using high-level model loading and automated export functions to reduce manual architecture definitions.

### Transfer Learning
Adapting a model trained on a large dataset (ImageNet) to a smaller, specific task (CIFAR-10).

### Fine-Tuning
Re-training a portion of a pre-trained model with a low learning rate to improve domain-specific performance.

### ONNX (Open Neural Network Exchange)
Exporting models with `dynamic_axes` to ensure hardware and framework agnosticism during inference.

---

# 📊 Models

## ResNet18 (Modified)

- **Architecture:** Pre-trained ResNet18 with a modified final `Linear` layer  
  `512 input nodes → 10 output nodes`
- **Activation:** ReLU (internal to ResNet)
- **Optimizer:** Adam (`lr = 0.0005` for fine-tuning)
- **Loss Function:** Cross-Entropy Loss (multi-class classification)

---

# 📈 Evaluation

## Task 1: Transfer Learning (Frozen Base)

- **Dataset:** CIFAR-10 (normalized and transformed)
- **Performance:** ~42% accuracy on the test set

---

## Task 2: Fine-Tuning (Unfrozen Layer 4)

- **Method:** Unfroze `layer4` and re-trained for 1 epoch
- **Performance:** ~67% accuracy on the test set

---

# ⚙️ Technical Topics

- `torchvision.models.resnet18` for model definition
- `torch.utils.data.DataLoader` with `batch_size = 32`
- `torch.onnx.export` with `dynamic_axes`
- `nbdime` for clean Jupyter Notebook version control
- Hardware detection logic for:
  - `cuda`
  - `mps`
  - `cpu`

---

# 🎯 Goal

Learn how to transition from building models from scratch to optimizing and deploying professional-grade architectures:

Standard PyTorch → Transfer Learning → Fine-Tuning → ONNX Deployment

This lecture prepares for:

- Efficient use of pre-trained models
- Optimizing model performance for specific datasets
- MLOps workflows and cross-platform model deployment

# 📘 Lecture 11 – Model Integration

In this lecture, I practiced the end-to-end process of model persistence and cross-platform compatibility. The focus was on serializing a trained model, exporting it to the industry-standard **ONNX** format, and verifying that predictions remain consistent across different runtime environments.

---

## ✅ What I Learned

- **Model Serialization:** Using `torch.save` and `torch.load` to persist model weights via `state_dict`.
- **Environment Agnostic Loading:** Mapping models to specific hardware (e.g., `map_location="cpu"`) during the loading process.
- **ONNX Inference:** Running models outside of PyTorch using the `onnxruntime` engine.
- **Verification Workflows:** Comparing raw PyTorch output shapes and values against ONNX runtime outputs to ensure parity.
- **Dynamic Input Handling:** Managing data transformations (normalization and tensor conversion) to match the expected input of a serialized model.
- **Sanity Checking:** Using random noise tensors (`torch.randn`) to verify model architecture integrity post-loading.

---

## 🧠 Key Concepts

- **Weights Persistence:** Understanding that `state_dict` only saves parameters, requiring the model architecture to be redefined before loading.
- **Serialization vs. Export:** Distinguishing between PyTorch-specific saving (`.pt`) and framework-interchangeable formats (`.onnx`).
- **Inference Sessions:** How `InferenceSession` optimizes the computational graph for faster execution during production.
- **Input/Output Mapping:** Correctly identifying input layer names (e.g., `_sess.get_inputs()[0].name`) when passing data to an ONNX runtime.

---

## 📊 Models

### ResNet18 (Integrated)
- **Source:** Pre-trained weights (adapted from Lecture 10).
- **Modification:** Final Fully Connected layer modified for **CIFAR-10** (10 output classes).
- **Format A:** PyTorch JIT/State Dict (`L11_model.pt`).
- **Format B:** ONNX Graph (`L11_model.onnx`).

---

## 📈 Evaluation

### Cross-Format Comparison
- **Dataset:** CIFAR-10 Test Set.
- **Consistency Check:** Verified that both the `.pt` file and the `.onnx` file produced identical output shapes `(batch_size, 10)`.
- **Validation:** Implemented a loop to pass real image data through both engines to verify prediction alignment.

---

## ⚙️ Technical Topics

- `pathlib.Path` for robust file system navigation.
- `torch.load` with `weights_only=True` for secure deserialization.
- `onnxruntime.InferenceSession` for high-performance deployment.
- `transforms.Compose` for matching preprocessing pipelines between training and inference.
- Manual evaluation mode toggle using `model.eval()` and `torch.no_grad()`.

---

## 🎯 Goal

Master the transition from "Training" to "Integration":

**Train/Fine-tune** → **Save State** → **Export to ONNX** → **Verify Consistency** → **Cross-Platform Readiness**

This lecture prepares for:
- Deploying models in non-Python environments (C++, C#, JavaScript).
- Standardizing model delivery for MLOps pipelines.
- Ensuring reproducible results across different hardware and frameworks.

-------------------------------------------------------------

# 📘 Lecture 12 – Docker & Containerization

In this lecture, I explored how to containerize machine learning applications to ensure they run consistently across different environments. I moved from running scripts locally to orchestrating a multi-service stack using **Docker** and **Docker Compose**.

## ✅ What I Learned

- **Writing Dockerfiles:** Creating custom images using `python:3.11-slim` and `nginx:alpine` as base layers.
- **Image Layering:** Understanding how `COPY`, `RUN`, and `WORKDIR` commands build up an immutable environment.
- **Service Orchestration:** Using Docker Compose to manage a three-component stack (**Frontend**, **Backend**, **Database**).
- **Network & Port Mapping:** Mapping internal container ports (e.g., 80, 8000) to host-accessible ports (e.g., 3000, 8000).
- **Environment Management:** Passing sensitive data and configuration like `DATABASE_URL` via environment variables.
- **Dependency Management:** Implementing `depends_on` and `healthcheck` to ensure the database is ready before the backend starts.
- **Persistent Storage:** Using Docker `volumes` to ensure PostgreSQL data persists even when containers are destroyed.

## 🧠 Key Concepts

- **Containerization:** Packaging code, runtime, and libraries into a single unit to solve the "it works on my machine" problem.
- **Microservices Architecture:** Separating the frontend (UI), backend (API/Logic), and database into independent, communicating services.
- **Infrastructure as Code (IaC):** Defining the entire infrastructure (networks, volumes, services) in a `docker-compose.yml` file.
- **Healthchecks:** Automating the monitoring of service readiness to prevent application crashes during startup.

## 📊 Services Stack

| Service | Technology | Role |
| :--- | :--- | :--- |
| **db** | PostgreSQL 15 | Persistent storage for application data. |
| **backend** | FastAPI (Python) | Serving the ML model and handling logic. |
| **frontend** | Nginx (HTML/CSS) | Simple user interface to interact with the API. |

## ⚙️ Technical Topics

- `FROM`, `COPY`, `RUN`, `CMD` (Dockerfile instructions)
- `docker compose up --build`
- `pg_isready` for database health monitoring
- Nginx static file serving
- Cross-container networking via service names

## 🎯 Goal

Transition from standalone scripts to a production-ready microservices environment:

**Script → Dockerfile → Image → Multi-container Stack → Orchestration**

This lecture prepares for:
- Deploying ML models to the cloud (AWS/Azure/GCP).
- Building scalable, multi-user web applications.
- Standardizing development environments for team collaboration.

-------------------------------------------------------------

# 📘 Lecture 13 – Future Trends & Ethics

In this final lecture, I explored the dual nature of modern AI: the drive for extreme computational efficiency using **JAX** and the essential responsibility of **AI Ethics**. The focus was on moving beyond standard training loops to understand high-performance numerical computing and the societal impact of model deployment.

## ✅ What I Learned

- **JAX Fundamentals:** Using `jax.numpy` (jnp) for hardware-accelerated numerical operations.
- **Just-In-Time (JIT) Compilation:** Using `jax.jit` to compile Python functions into optimized XLA kernels for massive speedups.
- **Automatic Differentiation:** Leveraging `jax.grad` to compute high-order derivatives automatically.
- **Performance Benchmarking:** Comparing pure NumPy execution against JIT-compiled versions to understand overhead and scaling.
- **AI Ethics Toolkits:** Familiarization with industry tools like **AIF360**, **Fairlearn**, and **Model Cards**.
- **Bias Mitigation:** Practicing bias audits on datasets and documenting collection consent.
- **Fairness Trade-offs:** Analyzing the tension between maximizing model accuracy and maintaining ethical fairness.

## 🧠 Key Concepts

- **XLA (Accelerated Linear Algebra):** The underlying compiler that JAX uses to fuse operations and minimize memory overhead.
- **The "Warm-up" Effect:** Understanding why the first run of a JIT-compiled function is slower (compilation time) while subsequent runs are "lightning fast."
- **Algorithmic Fairness:** The practice of ensuring model decisions do not disproportionately disadvantage specific groups.
- **Model Interpretability:** The challenge of explaining complex "black box" decisions to non-technical stakeholders.

## 📊 Performance & Tools

### JAX vs. NumPy
- **Task:** Matrix Multiplication and Trigonometric loops.
- **Observation:** JAX excels in high-repetition tasks where the logic can be mapped and executed on GPU/TPU.

### Ethical Frameworks
- **Audit Toolkits:** AIF360, What-If Tool.
- **Documentation:** Using Model Cards for transparency in data usage and performance limitations.

## ⚙️ Technical Topics

- `jax.jit` & `jax.grad`
- `jnp.array` (DeviceArray management)
- `block_until_ready()` (handling asynchronous execution in JAX)
- Bias audits and post-deployment monitoring
- Explanable AI (XAI) basics

## 🎯 Goal

Bridge the gap between high-performance computing and responsible AI deployment:

**Pure Math → JIT Compilation → Performance Scaling → Ethical Auditing → Responsible AI**

This final lecture concludes the journey by emphasizing that a "good" model is not just accurate and fast, but also fair and transparent.

-------------------------------------------------------------