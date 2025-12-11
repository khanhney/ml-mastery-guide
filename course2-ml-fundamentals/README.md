# Course 2: Machine Learning Fundamentals

> Master the core algorithms that form the foundation of all machine learning.

![Course 2 Roadmap](./Course%202.PNG)

---

## Table of Contents

1. [Linear Regression](#1-linear-regression)
2. [Polynomial Regression](#2-polynomial-regression)
3. [Logistic Regression](#3-logistic-regression)
4. [Naive Bayes](#4-naive-bayes)
5. [Support Vector Machine (SVM)](#5-support-vector-machine-svm)
6. [K-Nearest Neighbors (KNN)](#6-k-nearest-neighbors-knn)
7. [Decision Trees](#7-decision-trees)
8. [Model Selection & Hyperparameter Tuning](#8-model-selection--hyperparameter-tuning)

---

## 1. Linear Regression

Predicts continuous numerical values based on linear relationships between variables.

### 1.1 Simple Linear Regression

$$\hat{y} = \beta_0 + \beta_1 X$$

**Example after training:**

$$\hat{y} = 50000 + 1500X$$

$$\text{House Price} = 50,000 + 1,500 \times \text{Area (m}^2\text{)}$$

**Interpretation:** For each additional square meter, the house price increases by $1,500.

### 1.2 Multiple Linear Regression

$$\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \cdots + \beta_n X_n$$

**Compact notation:**
$$\hat{y} = \beta_0 + \sum_{i=1}^{n} \beta_i X_i$$

**Matrix form:**
$$\hat{y} = X\boldsymbol{\beta}$$

**Example after training:**

$$\hat{y} = 20000 + 1200X_1 + 5000X_2 - 800X_3$$

| Variable | Coefficient | Meaning |
|----------|-------------|---------|
| $\beta_0$ | 20,000 | Base price (intercept) |
| $\beta_1$ (Area) | 1,200 | Price increase per m² |
| $\beta_2$ (Bedrooms) | 5,000 | Price increase per bedroom |
| $\beta_3$ (Age) | -800 | Price decrease per year of age |

### 1.3 Industry Applications

| Industry | Use Case |
|----------|----------|
| Real Estate | Predicting house prices, car values, office rental prices |
| E-commerce | Forecasting sales revenue over time |
| Finance | Revenue prediction, cost estimation |

### 1.4 Key Concepts

- **Ordinary Least Squares (OLS):** Minimize sum of squared residuals
- **Assumptions:** Linearity, Independence, Homoscedasticity, Normality (LINE)
- **Metrics:** R-squared, Adjusted R-squared, MSE, RMSE, MAE

### 1.5 Regularization

| Type | Formula | Effect |
|------|---------|--------|
| **Ridge (L2)** | $\min \sum(y - \hat{y})^2 + \lambda\sum\beta_j^2$ | Shrinks coefficients |
| **Lasso (L1)** | $\min \sum(y - \hat{y})^2 + \lambda\sum|\beta_j|$ | Sparse coefficients (feature selection) |
| **ElasticNet** | Combination of L1 and L2 | Best of both |

### 1.6 Implementation

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Simple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# With Regularization
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
```

**When NOT to use:**
- Non-linear relationships
- Categorical target variable
- High multicollinearity without regularization

---

## 2. Polynomial Regression

Extension of linear regression for non-linear relationships.

### 2.1 Degree 2 (Quadratic)

$$\hat{y} = \beta_0 + \beta_1 X + \beta_2 X^2$$

**Example:**
$$\text{Revenue} = 100 + 5 \times \text{Ad Spend} - 0.02 \times \text{Ad Spend}^2$$

**Interpretation:** Diminishing returns - spending more on advertising eventually yields less incremental revenue.

### 2.2 General Polynomial (Degree n)

$$\hat{y} = \beta_0 + \sum_{i=1}^{n} \beta_i X^i$$

### 2.3 With Interaction Terms

$$\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1^2 + \beta_4 X_2^2 + \beta_5 X_1 X_2$$

**Implementation:**
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
model.fit(X_poly, y)
```

**Key Concepts:**
- Degree selection (avoid overfitting)
- Bias-variance tradeoff
- Use cross-validation to select degree

---

## 3. Logistic Regression

Binary/multiclass classification using sigmoid function for probability estimation.

### 3.1 Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 3.2 Binary Classification

**Probability of positive class:**

$$P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_n X_n)}}$$

**Compact form:**
$$P(y=1|X) = \sigma\left(\beta_0 + \sum_{i=1}^{n} \beta_i X_i\right)$$

### 3.3 Example: Customer Churn

$$P(\text{Churn}=1) = \frac{1}{1 + e^{-(-2.5 + 0.8X_1 - 1.2X_2 + 0.5X_3)}}$$

| Coefficient | Value | Interpretation |
|-------------|-------|----------------|
| $\beta_0$ | -2.5 | Base log-odds (intercept) |
| $\beta_1$ (Account age) | 0.8 | Older accounts → higher churn risk |
| $\beta_2$ (Transactions) | -1.2 | More transactions → lower churn risk |
| $\beta_3$ (Complaints) | 0.5 | More complaints → higher churn risk |

### 3.4 Log-Odds (Logit) Form

$$\log\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_n X_n$$

**Odds Ratio:** $e^{\beta_i}$ = how much odds multiply when $X_i$ increases by 1

### 3.5 Multiclass Classification (Softmax)

For $K$ classes:
$$P(y=k|X) = \frac{e^{\beta_{k0} + \sum_{i=1}^{n} \beta_{ki} X_i}}{\sum_{j=1}^{K} e^{\beta_{j0} + \sum_{i=1}^{n} \beta_{ji} X_i}}$$

### 3.6 Industry Applications

| Industry | Use Case |
|----------|----------|
| Fintech | Predicting customer churn, loan default |
| Healthcare | Disease risk prediction |
| Telecom | Classifying customers likely to renew |
| Marketing | Lead conversion prediction |

### 3.7 Implementation

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,                    # Inverse regularization strength
    penalty='l2',             # 'l1', 'l2', 'elasticnet'
    class_weight='balanced',  # Handle imbalanced data
    max_iter=1000
)
model.fit(X_train, y_train)

# Get probabilities
proba = model.predict_proba(X_test)
```

**When to use:**
- Need probability outputs
- Interpretable model required
- Baseline model for comparison

---

## 4. Naive Bayes

Probabilistic classifier based on Bayes' theorem with independence assumptions.

### 4.1 Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

For classification:
$$P(y|X) = \frac{P(X|y) \times P(y)}{P(X)}$$

### 4.2 "Naive" Assumption

Features are conditionally independent given the class:
$$P(X|y) = P(x_1|y) \times P(x_2|y) \times \cdots \times P(x_n|y)$$

### 4.3 Variants

| Type | Data Type | Use Case |
|------|-----------|----------|
| **GaussianNB** | Continuous (assumes normal distribution) | General purpose |
| **MultinomialNB** | Count data | Text classification |
| **BernoulliNB** | Binary features | Binary text features |
| **ComplementNB** | Count data | Imbalanced text data |

### 4.4 Industry Applications

| Industry | Use Case |
|----------|----------|
| Email Services | Automatic spam filtering |
| Customer Support | Classifying user intent |
| News | Document categorization |
| Social Media | Sentiment classification |

### 4.5 Implementation

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# For continuous features
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# For text classification
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts)
mnb = MultinomialNB()
mnb.fit(X_tfidf, y_train)
```

**Strengths:**
- Very fast training and prediction
- Works well with small datasets
- Handles high-dimensional data
- Good baseline for text classification

**Weaknesses:**
- Independence assumption rarely true
- Sensitive to feature correlations

---

## 5. Support Vector Machine (SVM)

Finds optimal hyperplane for classification with maximum margin.

### 5.1 Key Concepts

- **Maximum margin classifier:** Find hyperplane that maximizes distance to nearest points
- **Support vectors:** Data points closest to the decision boundary
- **Kernel trick:** Map to higher dimensions for non-linear boundaries

### 5.2 Kernels

| Kernel | Formula | Use Case |
|--------|---------|----------|
| **Linear** | $K(x, x') = x \cdot x'$ | Linearly separable data |
| **RBF** | $K(x, x') = e^{-\gamma||x-x'||^2}$ | Most common, general purpose |
| **Polynomial** | $K(x, x') = (x \cdot x' + c)^d$ | Polynomial decision boundary |

### 5.3 Industry Applications

| Industry | Use Case |
|----------|----------|
| Bioinformatics | Gene classification, protein structure |
| Finance | High-dimensional fraud detection |
| Image | Object detection, handwriting recognition |

### 5.4 Implementation

```python
from sklearn.svm import SVC, SVR

# Classification
svm = SVC(
    C=1.0,           # Regularization (higher = less regularization)
    kernel='rbf',    # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',   # Kernel coefficient
    class_weight='balanced'
)
svm.fit(X_train, y_train)
```

**When to use:**
- High-dimensional data
- Clear margin of separation
- Small to medium datasets

**Limitations:**
- Slow on large datasets
- Sensitive to feature scaling
- Difficult to interpret

---

## 6. K-Nearest Neighbors (KNN)

Instance-based learning using distance metrics.

### 6.1 Algorithm

1. Calculate distance to all training points
2. Select K nearest neighbors
3. Vote (classification) or average (regression)

### 6.2 Distance Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Euclidean** | $\sqrt{\sum(x_i - y_i)^2}$ | Continuous features |
| **Manhattan** | $\sum|x_i - y_i|$ | High-dimensional data |
| **Cosine** | $1 - \frac{x \cdot y}{||x|| \cdot ||y||}$ | Text, sparse data |
| **Minkowski** | $(\sum|x_i - y_i|^p)^{1/p}$ | Generalized |

### 6.3 Industry Applications

| Industry | Use Case |
|----------|----------|
| Recommendation | Item similarity matching |
| Anomaly Detection | Identifying outliers |
| Healthcare | Patient similarity analysis |

### 6.4 Implementation

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,      # K value (odd for binary classification)
    weights='distance', # 'uniform' or 'distance'
    metric='euclidean'
)
knn.fit(X_train, y_train)
```

**Key Considerations:**
- **Feature scaling is CRITICAL**
- K selection: Use cross-validation
- Odd K for binary classification

**Limitations:**
- Slow prediction on large datasets
- Curse of dimensionality
- Sensitive to irrelevant features

---

## 7. Decision Trees

Tree-based model that makes decisions by splitting data on feature values.

### 7.1 Splitting Criteria

**Gini Impurity:**
$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

**Entropy:**
$$Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

**Information Gain:**
$$IG = Entropy_{parent} - \sum \frac{n_{child}}{n_{parent}} \times Entropy_{child}$$

### 7.2 Industry Applications

| Industry | Use Case |
|----------|----------|
| Healthcare | Diagnosing diseases based on symptoms |
| HR Tech | Supporting hiring/recruitment decisions |
| Banking | Credit approval rules |
| Insurance | Risk assessment |

### 7.3 Implementation

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

dt = DecisionTreeClassifier(
    max_depth=5,              # Maximum tree depth
    min_samples_split=10,     # Min samples to split
    min_samples_leaf=5,       # Min samples in leaf
    criterion='gini'          # 'gini' or 'entropy'
)
dt.fit(X_train, y_train)

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, filled=True)
plt.show()
```

### 7.4 Hyperparameters

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `max_depth` | Maximum tree depth | Higher = more overfitting |
| `min_samples_split` | Min samples to split | Higher = more regularization |
| `min_samples_leaf` | Min samples in leaf | Higher = smoother predictions |
| `criterion` | Splitting criterion | 'gini' faster, 'entropy' slightly better |

**Strengths:**
- Highly interpretable (can visualize)
- No feature scaling needed
- Handles non-linear relationships
- Built-in feature importance

**Weaknesses:**
- Prone to overfitting
- Unstable (small data changes → different tree)
- Biased toward features with many levels

---

## 8. Model Selection & Hyperparameter Tuning

### 8.1 Industry Applications

| Industry | Use Case |
|----------|----------|
| AI Ops | Selecting most efficient model for production |
| Data Science | Comparing models by accuracy, recall |

### 8.2 Hyperparameter Tuning Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Grid Search** | Exhaustive search | Thorough | Slow |
| **Random Search** | Random sampling | Faster, often better | May miss optimal |
| **Bayesian Optimization** | Probabilistic model-guided | Efficient | Complex setup |
| **Optuna** | Advanced framework | Auto-pruning | Learning curve |

### 8.3 Implementation

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

### 8.4 Model Comparison Framework

```
1. Establish Baseline (simple model)
2. Cross-validate multiple algorithms
3. Select top 2-3 candidates
4. Tune hyperparameters
5. Final evaluation on hold-out test set
```

---

## Algorithm Comparison Summary

| Algorithm | Type | Interpretable | Scaling Needed | Speed | Best For |
|-----------|------|---------------|----------------|-------|----------|
| Linear Regression | Regression | High | Yes | Fast | Linear relationships |
| Logistic Regression | Classification | High | Yes | Fast | Probability outputs |
| Naive Bayes | Classification | Medium | No | Very Fast | Text, baseline |
| SVM | Both | Low | Yes | Slow | High-dim, small data |
| KNN | Both | Medium | Yes | Slow (predict) | Similarity-based |
| Decision Tree | Both | High | No | Fast | Interpretability |

---

## Practice Projects

| Algorithm | Project | Dataset |
|-----------|---------|---------|
| Linear Regression | House price prediction | Kaggle Housing |
| Logistic Regression | Titanic survival | Kaggle Titanic |
| Naive Bayes | SMS spam classifier | UCI SMS Spam |
| SVM | Handwriting recognition | MNIST |
| KNN | Iris classification | Sklearn Iris |
| Decision Tree | Loan approval | Kaggle Lending Club |

---

[← Back to Course 1](../course1-statistics-essentials/README.md) | [Main Roadmap](../README.md) | [Next: Course 3 - ML Advanced →](../course3-ml-advanced/README.md)
