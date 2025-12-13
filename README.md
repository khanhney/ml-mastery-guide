<div align="center">

# ML Mastery Guide

### The Complete Machine Learning & Deep Learning Roadmap

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**From Zero to Production-Ready ML Engineer**

*Built with the 80/20 principle: Focus on the 20% that delivers 80% of results*

</div>

---

## Overview

A comprehensive, practical roadmap for mastering Machine Learning and Deep Learning. Each course includes mathematical formulas, code examples, and real-world use cases.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              ML MASTERY LEARNING PATH                                         │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                               │
│   Course 1           Course 2           Course 3           Course 4           Course 5       │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐    │
│  │Statistics│  ───► │   ML     │  ───► │   ML     │  ───► │  Deep    │  ───► │  NLP &   │    │
│  │Essentials│       │Fundamentals      │ Advanced │       │ Learning │       │  LLMs    │    │
│  └──────────┘       └──────────┘       └──────────┘       └──────────┘       └──────────┘    │
│                                                                                               │
│   - Data Prep        - Linear Reg       - Random Forest    - NN              - Tokenization  │
│   - Feature Eng      - Logistic Reg     - XGBoost/LightGBM - CNN             - Transformers  │
│   - Evaluation       - Naive Bayes      - Clustering       - RNN/LSTM        - BERT/GPT      │
│   - Best Practices   - SVM, KNN, Trees  - PCA, Stacking    - Transfer        - RAG/Agents    │
│                                                                                               │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Course Structure

### [Course 1: Statistics Essentials & Data Foundations](./course1-statistics-essentials/README.md)

> **Foundation skills that make or break your ML models**

| Topic | Key Concepts |
|-------|--------------|
| **Data Cleaning** | Missing values (MCAR, MAR, MNAR), Outliers, Duplicates |
| **Feature Engineering** | Encoding, Scaling, Transformation, Binning |
| **Time-Based Features** | Lag features, Rolling statistics, Cyclical encoding |
| **Imbalanced Data** | SMOTE, ADASYN, Class weights, Threshold tuning |
| **Feature Selection** | Correlation, RFE, L1 regularization, Feature importance |
| **Model Evaluation** | Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC |
| **Cross-Validation** | K-Fold, Stratified, Time Series Split |
| **Common Pitfalls** | Data leakage, Overfitting, Wrong metrics |

---

### [Course 2: Machine Learning Fundamentals](./course2-ml-fundamentals/README.md)

> **Core algorithms that form the foundation of ML**

| Algorithm | Type | Industry Use Cases |
|-----------|------|-------------------|
| **Linear Regression** | Regression | Real Estate pricing, Sales forecasting |
| **Logistic Regression** | Classification | Churn prediction, Loan default |
| **Naive Bayes** | Classification | Spam filtering, Intent classification |
| **SVM** | Both | Gene classification, Fraud detection |
| **KNN** | Both | Recommendation, Anomaly detection |
| **Decision Trees** | Both | Disease diagnosis, Credit approval |

**Key Skills:** OLS, Sigmoid, Bayes theorem, Kernel trick, Distance metrics, Gini/Entropy

---

### [Course 3: Machine Learning Advanced](./course3-ml-advanced/README.md)

> **Production-ready algorithms that win competitions**

| Topic | Algorithms | Industry Use Cases |
|-------|------------|-------------------|
| **Bagging** | Random Forest, Extra Trees | Fraud detection, Pathology analysis |
| **Boosting** | XGBoost, LightGBM, CatBoost | Credit scoring, CTR prediction |
| **Stacking** | Meta-learning, Blending | Competition winning, Ensemble |
| **Clustering** | K-Means, DBSCAN, HDBSCAN | Customer segmentation, Marketing |
| **Dimensionality** | PCA, t-SNE, UMAP | Sensor data, Image compression |
| **MLOps** | MLflow, FastAPI, Monitoring | Production deployment |

**Key Skills:** Bootstrap aggregating, Gradient boosting, Regularization, Early stopping

---

### [Course 4: Deep Learning](./course4-deep-learning/README.md)

> **Neural networks for complex pattern recognition**

| Architecture | Type | Industry Use Cases |
|--------------|------|-------------------|
| **Neural Networks** | Tabular | Employee attrition, Risk modeling |
| **CNN** | Image | Face detection, Medical imaging (CT, X-ray) |
| **RNN/LSTM** | Sequence | Stock prediction, Sentiment analysis |
| **Transfer Learning** | Image/Text | Any image task with limited data |

**Key Skills:** Backpropagation, Activation functions, Dropout, Convolution, Attention

---

### [Course 5: NLP & Large Language Models](./course5-nlp-llm/README.md)

> **From classical NLP to modern Transformers and LLMs**

| Topic | Techniques | Industry Use Cases |
|-------|------------|-------------------|
| **Lexical Processing** | Tokenization, Stemming, TF-IDF | Search engines, Spam filtering |
| **Syntactic Processing** | POS Tagging, Parsing, NER | Grammar checkers, Information extraction |
| **Semantic Processing** | Word2Vec, GloVe, FastText | Semantic search, Recommendation |
| **Transformers** | Self-Attention, Multi-Head Attention | Machine translation, Text generation |
| **LLMs** | BERT, GPT, LLaMA, Fine-tuning | Chatbots, Code generation, QA systems |
| **Applications** | RAG, Agents, Prompt Engineering | Enterprise AI, Knowledge assistants |

**Key Skills:** Attention mechanism, Positional encoding, RLHF, LoRA, Prompt engineering

---

## Quick Reference

### Algorithm Selection by Problem Type

| Problem | First Try | Then Try | Advanced |
|---------|-----------|----------|----------|
| **Tabular Classification** | XGBoost | LightGBM, CatBoost | Neural Network |
| **Tabular Regression** | XGBoost | LightGBM | Neural Network |
| **Image Classification** | Transfer Learning | EfficientNet | Custom CNN |
| **Text Classification** | TF-IDF + LogReg | BERT | Fine-tuned LLM |
| **Time Series** | Prophet, ARIMA | LSTM | Transformer |
| **Clustering** | K-Means | HDBSCAN | GMM |
| **Anomaly Detection** | Isolation Forest | One-class SVM | Autoencoder |

### Algorithm Selection by Data Size

| Data Size | Recommended Approach |
|-----------|---------------------|
| < 1,000 | Simple models, strong regularization |
| 1,000 - 100,000 | Gradient boosting (XGBoost, LightGBM) |
| 100,000 - 1M | Gradient boosting or Neural Networks |
| > 1M | Neural Networks, LightGBM |

---

## Learning Path

```
Phase 1                  Phase 2                  Phase 3
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ Course 1           │──►│ Course 2           │──►│ Course 3           │
│ Data + Evaluation  │   │ Core ML Algorithms │   │ Ensemble + MLOps   │
└────────────────────┘   └────────────────────┘   └────────────────────┘
                                                           │
                         ┌─────────────────────────────────┘
                         ▼
Phase 4                  Phase 5
┌────────────────────┐   ┌────────────────────┐
│ Course 4           │──►│ Course 5           │
│ Deep Learning      │   │ NLP & LLMs         │
└────────────────────┘   └────────────────────┘
```

---

## Prerequisites

| Level | Required Knowledge |
|-------|-------------------|
| **Course 1** | Python basics, NumPy, Pandas |
| **Course 2** | Course 1 + Basic statistics |
| **Course 3** | Course 2 + Scikit-learn |
| **Course 4** | Course 3 + Linear algebra basics |
| **Course 5** | Course 4 + PyTorch/TensorFlow basics |

---

## Tools & Libraries

| Category | Tools |
|----------|-------|
| Data Processing | Pandas, NumPy, Polars |
| Classical ML | Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Deep Learning | TensorFlow/Keras, PyTorch |
| NLP | Hugging Face Transformers, spaCy, NLTK |
| LLM Development | LangChain, LlamaIndex, OpenAI API |
| Experiment Tracking | MLflow, Weights & Biases |
| Model Serving | FastAPI, BentoML, vLLM |

---

## Practice Projects

| Course | Project | Dataset |
|--------|---------|---------|
| Course 1 | Data cleaning + EDA | Titanic |
| Course 2 | Classification pipeline | Titanic, Iris |
| Course 3 | Ensemble + Hyperparameter tuning | Customer Churn |
| Course 4 | Image classification | CIFAR-10, ChestX-ray |
| Course 5 | Text classification + Chatbot | IMDB, Custom QA |

---

## File Structure

```
ml-mastery-guide/
├── README.md                          # This file (overview)
├── course1-statistics-essentials/
│   └── README.md                      # Data prep, evaluation, pitfalls
├── course2-ml-fundamentals/
│   └── README.md                      # Linear, Logistic, NB, SVM, KNN, Trees
├── course3-ml-advanced/
│   └── README.md                      # RF, XGBoost, Clustering, PCA, MLOps
├── course4-deep-learning/
│   └── README.md                      # NN, CNN, RNN, Transfer Learning
└── course5-nlp-llm/
    └── README.md                      # NLP, Transformers, LLMs, RAG, Agents
```

---

## Contributing

Feel free to contribute by:
1. Adding new use cases
2. Updating resources
3. Sharing your learning experience
4. Reporting errors or suggesting improvements

---

## License

This roadmap is open-source and available for educational purposes (MIT License).

---

<div align="center">

**Happy Learning!**

*Remember: The best model is the one that solves the business problem, not the one with the highest accuracy.*

</div>
