
# 🧠 Logistic Regression from Scratch using Python + NumPy

This project demonstrates how to **build and train a Logistic Regression model from scratch** — using only **Python** and **NumPy**, without relying on any machine learning libraries like scikit-learn.

The goal is to understand **how Logistic Regression actually works behind the scenes** — mathematically and programmatically.

---

## 📘 Overview

Logistic Regression is a **classification algorithm** used to predict binary outcomes (e.g., 0 or 1, Yes or No, True or False).

It works by modeling the probability that a given input belongs to a particular class using the **sigmoid function**.

### 🔹 Sigmoid Function

The sigmoid function converts any real number into a value between 0 and 1:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

---

## ⚙️ Implementation Steps

### 1️⃣ Create the LogisticRegression Class
The class contains three main methods:

- `__init__()` → Initialize hyperparameters (learning rate, iterations).
- `fit()` → Train the model using **Gradient Descent**.
- `predict()` → Predict class labels for new samples.
- `_sigmoid()` → Apply the sigmoid activation function.

### 2️⃣ Gradient Descent
During training, the model adjusts its **weights** and **bias** to minimize the prediction error.

At each step:
- Calculate the predicted probabilities using the sigmoid function.
- Compute gradients of the loss function.
- Update parameters in the opposite direction of the gradient.

This process repeats until convergence.

---

## 🧩 Example Usage

```python
import numpy as np

# Example dataset (binary classification)
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7]
])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize and train model
model = LogisticRegression(lr=0.1, n_iters=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[2, 3], [5, 6]]))
print("Predictions:", predictions)
