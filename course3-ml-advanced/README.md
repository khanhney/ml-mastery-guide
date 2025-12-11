# Course 3: Machine Learning Advanced

> **80/20 Insight:** Ensemble methods, especially XGBoost/LightGBM, win the majority of Kaggle competitions and are the go-to choice for tabular data in production.

![Course 3 Roadmap](./Course%203.PNG)

---

## Table of Contents

1. [Ensemble Methods Overview](#1-ensemble-methods-overview)
2. [Bagging & Random Forest](#2-bagging--random-forest)
3. [Boosting Methods](#3-boosting-methods)
4. [Stacking & Blending](#4-stacking--blending)
5. [Clustering](#5-clustering)
6. [Dimensionality Reduction (PCA)](#6-dimensionality-reduction-pca)
7. [MLOps Fundamentals](#7-mlops-fundamentals)

---

## 1. Ensemble Methods Overview

Ensemble methods combine multiple models to produce better predictions than any single model.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENSEMBLE METHODS TAXONOMY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              Ensemble Methods                                │
│                                     │                                        │
│          ┌──────────────────────────┼──────────────────────────┐            │
│          ▼                          ▼                          ▼            │
│    ┌──────────┐              ┌──────────┐              ┌──────────┐         │
│    │ BAGGING  │              │ BOOSTING │              │ STACKING │         │
│    └──────────┘              └──────────┘              └──────────┘         │
│          │                          │                          │            │
│          ▼                          ▼                          ▼            │
│    Parallel Training          Sequential Training      Meta-Learning        │
│    Reduce Variance            Reduce Bias              Combine Strengths    │
│          │                          │                          │            │
│          ▼                          ▼                          ▼            │
│    - Random Forest            - AdaBoost               - Blending           │
│    - Bagged Trees             - Gradient Boosting      - Meta-classifier    │
│    - Extra Trees              - XGBoost                - Multi-layer        │
│                               - LightGBM                                    │
│                               - CatBoost                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Comparison Summary:**

| Method | Training | Reduces | Base Models | Parallelizable |
|--------|----------|---------|-------------|----------------|
| **Bagging** | Parallel | Variance | Homogeneous | Yes |
| **Boosting** | Sequential | Bias | Homogeneous | Limited |
| **Stacking** | Parallel + Meta | Both | Heterogeneous | Yes (base) |

---

## 2. Bagging & Random Forest

### 2.1 Bagging (Bootstrap Aggregating)

**Core Idea:** Train multiple models on different random subsets of data, then aggregate predictions.

```
┌─────────────────────────────────────────────────────────────────┐
│                    BAGGING MECHANISM                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│     Original Dataset ──► Bootstrap Sampling (with replacement)  │
│                                    │                             │
│          ┌─────────────────────────┼─────────────────────────┐  │
│          ▼                         ▼                         ▼  │
│     ┌─────────┐              ┌─────────┐              ┌─────────┐│
│     │ Model 1 │              │ Model 2 │              │ Model N ││
│     └─────────┘              └─────────┘              └─────────┘│
│          │                         │                         │  │
│          └─────────────────────────┼─────────────────────────┘  │
│                                    ▼                             │
│                           Aggregate (Vote/Mean)                  │
│                                    │                             │
│                                    ▼                             │
│                            Final Prediction                      │
└─────────────────────────────────────────────────────────────────┘
```

**Mathematical Formulation:**

For regression (averaging):
$$\hat{y}_{bagging} = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(x)$$

For classification (majority voting):
$$\hat{y}_{bagging} = \text{mode}\{\hat{y}_1(x), \hat{y}_2(x), ..., \hat{y}_B(x)\}$$

**Implementation:**
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
print(f"OOB Score: {bagging.oob_score_:.4f}")
```

### 2.2 Random Forest

**Random Forest = Bagging + Random Feature Selection**

| Industry | Use Case |
|----------|----------|
| Finance | Detecting fraudulent banking transactions |
| Healthcare | Analyzing pathology from clinical data |
| E-commerce | Product recommendation |
| Insurance | Claim prediction |

**Key Innovation:** At each split, select random subset of √p features (decorrelates trees)

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    max_features='sqrt',
    min_samples_split=5,
    min_samples_leaf=2,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# Feature Importance
import pandas as pd
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

**Hyperparameters:**

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_estimators` | Number of trees | 100-1000 |
| `max_depth` | Tree depth | None, 10-30 |
| `max_features` | Features per split | 'sqrt', 'log2' |
| `min_samples_split` | Min samples to split | 2-20 |

---

## 3. Boosting Methods

**Core Idea:** Train models sequentially, each correcting errors of previous models.

$$F_M(x) = \sum_{m=1}^{M} \gamma_m h_m(x)$$

### 3.1 AdaBoost

Adjusts sample weights based on classification errors.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
```

### 3.2 Gradient Boosting

Fits new models to negative gradients (residuals).

```python
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
```

### 3.3 XGBoost (eXtreme Gradient Boosting)

**Industry standard.** Optimized with regularization.

| Industry | Use Case |
|----------|----------|
| Banking | Customer credit scoring |
| Insurance | Detecting fraudulent claims |
| E-commerce | Click-through rate prediction |

**Objective Function:**
$$Obj = \sum L(y_i, \hat{y}_i) + \sum \Omega(f_k)$$

Where regularization: $\Omega(f) = \gamma T + \frac{1}{2}\lambda||w||^2$

```python
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method='hist',
    early_stopping_rounds=50,
    random_state=42
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)
```

### 3.4 LightGBM

**Fastest boosting library.** Uses leaf-wise growth.

```
XGBoost (Level-wise):          LightGBM (Leaf-wise):
       [Root]                        [Root]
      /      \                      /      \
   [L1]      [L1]                [L1]    [Best]
  /    \    /    \                      /    \
[L2]  [L2][L2]  [L2]                 [L2]  [Best]
```

```python
import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,          # Main complexity control
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)]
)
```

### 3.5 CatBoost

**Best for categorical features.** Uses ordered boosting.

```python
from catboost import CatBoostClassifier

cat_features = ['gender', 'city', 'device_type']

cat_clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    auto_class_weights='Balanced',
    early_stopping_rounds=50,
    verbose=100
)

cat_clf.fit(X_train, y_train, eval_set=(X_val, y_val))
```

### 3.6 Boosting Comparison

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Speed** | Fast | Fastest | Medium |
| **Memory** | Medium | Low | High |
| **Categorical** | Encoding needed | Native | Best native |
| **GPU Support** | Yes | Yes | Yes |
| **Best For** | General purpose | Large data | Many categoricals |

**Decision Guide:**
```
Many categorical features? ──► CatBoost
Very large dataset (>1M)? ──► LightGBM
Need minimal deps? ──► HistGradientBoosting
Default choice ──► XGBoost
```

---

## 4. Stacking & Blending

### 4.1 Stacking

Train multiple diverse models, then train a meta-model on their predictions.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STACKING ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                        Training Data                             │
│          ┌──────────────────┼──────────────────┐                │
│          ▼                  ▼                  ▼                │
│    ┌───────────┐     ┌───────────┐     ┌───────────┐           │
│    │  XGBoost  │     │ LightGBM  │     │ CatBoost  │  Level 0  │
│    └───────────┘     └───────────┘     └───────────┘           │
│          │                  │                  │                │
│          └──────────────────┼──────────────────┘                │
│                             ▼                                    │
│                   ┌─────────────────┐                           │
│                   │   Meta-Model    │  Level 1                  │
│                   │ (Logistic Reg)  │                           │
│                   └─────────────────┘                           │
│                             │                                    │
│                             ▼                                    │
│                    Final Prediction                              │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('xgb', xgb.XGBClassifier(n_estimators=100)),
    ('lgb', lgb.LGBMClassifier(n_estimators=100))
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)
stacking.fit(X_train, y_train)
```

### 4.2 Blending

Simpler alternative using holdout set instead of cross-validation.

```python
# Split for blending
X_build, X_blend, y_build, y_blend = train_test_split(X_train, y_train, test_size=0.2)

# Train base models
models = {
    'xgb': xgb.XGBClassifier().fit(X_build, y_build),
    'lgb': lgb.LGBMClassifier().fit(X_build, y_build)
}

# Generate blend predictions
blend_preds = pd.DataFrame({
    name: model.predict_proba(X_blend)[:, 1]
    for name, model in models.items()
})

# Train meta-model
meta = LogisticRegression()
meta.fit(blend_preds, y_blend)
```

### 4.3 Voting Ensemble

```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', xgb.XGBClassifier(n_estimators=100)),
        ('lgb', lgb.LGBMClassifier(n_estimators=100))
    ],
    voting='soft',
    weights=[1, 2, 2]
)
```

---

## 5. Clustering

Grouping similar data points without labeled data.

### 5.1 Industry Applications

| Industry | Use Case |
|----------|----------|
| E-commerce | Customer segmentation |
| Retail | Building targeted marketing campaigns |
| Biology | Gene expression clustering |
| Social Media | Community detection |

### 5.2 Algorithm Comparison

| Algorithm | Strengths | Weaknesses | Best For |
|-----------|-----------|------------|----------|
| **K-Means** | Fast, scalable | Spherical clusters, needs K | Large data |
| **Hierarchical** | No K needed, dendrogram | Slow on large data | Small data |
| **DBSCAN** | Arbitrary shapes, handles outliers | Sensitive to params | Irregular clusters |
| **HDBSCAN** | Robust DBSCAN variant | Slower | Unknown K |

### 5.3 K-Means Implementation

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Finding optimal K
scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    scores.append(silhouette_score(X, kmeans.labels_))

# Plot elbow curve
import matplotlib.pyplot as plt
plt.plot(range(2, 11), scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Final model
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X)
```

### 5.4 DBSCAN

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.5,           # Maximum distance between samples
    min_samples=5,     # Minimum samples in neighborhood
    metric='euclidean'
)
clusters = dbscan.fit_predict(X)

# -1 labels are outliers
n_outliers = (clusters == -1).sum()
```

---

## 6. Dimensionality Reduction (PCA)

Principal Component Analysis - preserving maximum variance.

### 6.1 Industry Applications

| Industry | Use Case |
|----------|----------|
| IoT | Reducing sensor data dimensions for faster training |
| Image Processing | Image compression before classification |
| Finance | Portfolio risk factors |
| Genomics | Gene expression analysis |

### 6.2 Key Concepts

- **Eigenvalues:** Variance explained by each component
- **Eigenvectors:** Direction of maximum variance
- **Variance explained ratio:** How much info each component captures

### 6.3 Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale data first (critical for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Keep 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"Original features: {X.shape[1]}")
print(f"Reduced features: {X_pca.shape[1]}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Scree plot
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_.cumsum(), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()
```

### 6.4 Other Methods

| Method | Type | Best For |
|--------|------|----------|
| **PCA** | Linear | General purpose |
| **t-SNE** | Non-linear | Visualization (2D/3D) |
| **UMAP** | Non-linear | Visualization + preserves structure |
| **LDA** | Supervised | Classification with dim reduction |

---

## 7. MLOps Fundamentals

> **80/20 Insight:** 87% of ML models never make it to production. MLOps skills bridge the gap.

### 7.1 ML Project Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML PROJECT LIFECYCLE                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Problem  │───▶│   Data   │───▶│  Model   │───▶│  Deploy  │  │
│  │ Framing  │    │   Prep   │    │   Dev    │    │ Monitor  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │              │               │               │          │
│       └──────────────┴───────────────┴───────────────┘          │
│                         ITERATE                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Experiment Tracking

```python
import mlflow

mlflow.set_experiment("customer_churn")

with mlflow.start_run():
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("max_depth", 5)

    model = XGBClassifier(max_depth=5)
    model.fit(X_train, y_train)

    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
    mlflow.sklearn.log_model(model, "model")
```

### 7.3 Model Serving

```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: dict):
    X = preprocess(features)
    prediction = model.predict([X])
    return {"prediction": int(prediction[0])}
```

### 7.4 Model Monitoring

**What to Monitor:**
- Performance metrics (accuracy, latency)
- Data drift (feature distributions)
- Model drift (prediction distributions)

---

## Algorithm Selection Quick Guide

| Problem | First Try | Then Try |
|---------|-----------|----------|
| Tabular Classification | XGBoost | LightGBM, CatBoost |
| Tabular Regression | XGBoost | LightGBM |
| Customer Segmentation | K-Means | HDBSCAN |
| Dimensionality Reduction | PCA | UMAP |

---

## Practice Projects

| Algorithm | Project | Dataset |
|-----------|---------|---------|
| Random Forest | Credit card fraud | Kaggle Fraud |
| XGBoost | Customer churn | Telco Churn |
| Clustering | Customer segmentation | UCI Retail |
| PCA | Image compression | MNIST |

---

[← Back to Course 2](../course2-ml-fundamentals/README.md) | [Main Roadmap](../README.md) | [Next: Course 4 - Deep Learning →](../course4-deep-learning/README.md)
