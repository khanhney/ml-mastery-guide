# Course 1: Statistics Essentials

> **Foundation skills that form the backbone of Data Science and Machine Learning**

![Course 1 Roadmap](./Course%201.PNG)

---

## Table of Contents

1. [Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
2. [Descriptive Statistics & Probability](#2-descriptive-statistics--probability)
3. [Central Limit Theorem (CLT)](#3-central-limit-theorem-clt)
4. [Hypothesis Testing](#4-hypothesis-testing)
5. [SQL & Data Management](#5-sql--data-management)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Evaluation Metrics](#7-model-evaluation-metrics)
8. [Common Pitfalls & Best Practices](#8-common-pitfalls--best-practices)
9. [Assignment / Project](#9-assignment--project)

---

## 1. Exploratory Data Analysis (EDA)

> **80/20 Insight:** Good EDA reveals 80% of the insights before any modeling begins.

### 1.1 Data Sourcing

| Source Type | Examples | Considerations |
|-------------|----------|----------------|
| **Public Data** | Kaggle, UCI ML, Government portals | Free, well-documented |
| **Private Data** | Company databases, APIs | Access control, privacy |
| **Web Scraping** | BeautifulSoup, Selenium | Legal considerations |
| **APIs** | Twitter, Google, Financial | Rate limits, authentication |

### 1.2 Data Cleaning

| Task | Techniques | Tools |
|------|------------|-------|
| **Missing Values** | Mean/Median imputation, KNN Imputer, Forward/Backward fill | `sklearn.impute`, Pandas |
| **Outliers** | IQR method, Z-score, Isolation Forest | NumPy, Sklearn |
| **Duplicates** | Exact match, Fuzzy matching | Pandas, `fuzzywuzzy` |
| **Data Types** | Type conversion, Datetime parsing | Pandas |
| **Inconsistent Data** | Standardization, Regex cleaning | Pandas, re |

**Missing Data Types:**
| Type | Description | Strategy |
|------|-------------|----------|
| **MCAR** | Missing Completely At Random | Safe to drop or impute |
| **MAR** | Missing At Random (depends on other features) | Use predictive imputation |
| **MNAR** | Missing Not At Random | Requires domain knowledge |

```python
import pandas as pd
import numpy as np

# Check missing values
df.isnull().sum()

# Imputation strategies
from sklearn.impute import SimpleImputer, KNNImputer

# Mean/Median imputation
imputer = SimpleImputer(strategy='median')
df['column'] = imputer.fit_transform(df[['column']])

# KNN imputation (considers relationships)
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = knn_imputer.fit_transform(df)
```

### 1.3 Univariate Analysis

Analysis of single variables.

| Variable Type | Visualizations | Metrics |
|---------------|----------------|---------|
| **Numerical** | Histogram, Box plot, KDE | Mean, Median, Std, Skewness |
| **Categorical** | Bar chart, Pie chart | Frequency, Mode, Cardinality |

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Numerical variable
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df['price'].hist(ax=axes[0])
df['price'].plot.box(ax=axes[1])
sns.kdeplot(df['price'], ax=axes[2])

# Categorical variable
df['category'].value_counts().plot.bar()
```

### 1.4 Bivariate Analysis

Analysis of relationships between two variables.

| Combination | Visualization | Metric |
|-------------|---------------|--------|
| **Num vs Num** | Scatter plot, Heatmap | Correlation coefficient |
| **Cat vs Num** | Box plot, Violin plot | Group means, ANOVA |
| **Cat vs Cat** | Stacked bar, Heatmap | Chi-square, Cramér's V |

```python
# Correlation matrix
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')

# Scatter plot with regression line
sns.regplot(x='feature1', y='target', data=df)

# Box plot by category
sns.boxplot(x='category', y='price', data=df)
```

### 1.5 Derived Metrics

Creating new meaningful metrics from existing data.

| Derived Metric | Formula | Use Case |
|----------------|---------|----------|
| **Growth Rate** | $(V_{new} - V_{old}) / V_{old}$ | Sales trends |
| **Conversion Rate** | Conversions / Total visits | Marketing |
| **Customer Lifetime Value** | Avg purchase × Frequency × Lifespan | CRM |
| **Churn Rate** | Lost customers / Total customers | Retention |

```python
# Examples of derived metrics
df['revenue_per_customer'] = df['total_revenue'] / df['num_customers']
df['profit_margin'] = (df['revenue'] - df['cost']) / df['revenue']
df['yoy_growth'] = df['sales'].pct_change(periods=12)
```

---

## 2. Descriptive Statistics & Probability

### 2.1 Descriptive Statistics

| Measure | Type | Formula | Use |
|---------|------|---------|-----|
| **Mean** | Central tendency | $\bar{x} = \frac{\sum x_i}{n}$ | Average value |
| **Median** | Central tendency | Middle value | Robust to outliers |
| **Mode** | Central tendency | Most frequent | Categorical data |
| **Variance** | Dispersion | $\sigma^2 = \frac{\sum(x_i - \bar{x})^2}{n}$ | Spread |
| **Std Dev** | Dispersion | $\sigma = \sqrt{\sigma^2}$ | Same unit as data |
| **Skewness** | Shape | Asymmetry measure | Distribution shape |
| **Kurtosis** | Shape | Tail heaviness | Outlier presence |

```python
# All descriptive stats at once
df.describe()

# Additional stats
from scipy import stats
print(f"Skewness: {stats.skew(df['column'])}")
print(f"Kurtosis: {stats.kurtosis(df['column'])}")
```

### 2.2 Discrete Probability Distributions

| Distribution | Formula | Parameters | Use Case |
|--------------|---------|------------|----------|
| **Bernoulli** | $P(X=k) = p^k(1-p)^{1-k}$ | p (success prob) | Single trial (coin flip) |
| **Binomial** | $P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$ | n (trials), p (prob) | Number of successes in n trials |
| **Poisson** | $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$ | λ (rate) | Events in fixed interval |

**Bernoulli Distribution:**
- Single trial with two outcomes (success/failure)
- Example: Email opened (1) or not (0)

```python
from scipy import stats

# Bernoulli
bernoulli = stats.bernoulli(p=0.3)
print(f"P(X=1): {bernoulli.pmf(1)}")  # 0.3
```

**Binomial Distribution:**
- Number of successes in n independent trials
- Example: Number of customers who buy out of 100 visitors

```python
# Binomial: 100 trials, 30% success rate
binom = stats.binom(n=100, p=0.3)
print(f"P(X=30): {binom.pmf(30):.4f}")
print(f"P(X≤25): {binom.cdf(25):.4f}")
print(f"Expected: {binom.mean()}")  # 30
```

**Poisson Distribution:**
- Events occurring in fixed interval
- Example: Number of website visits per hour

```python
# Poisson: average 5 events per interval
poisson = stats.poisson(mu=5)
print(f"P(X=3): {poisson.pmf(3):.4f}")
print(f"P(X≤3): {poisson.cdf(3):.4f}")
```

### 2.3 Continuous Probability Distributions

| Distribution | PDF | Parameters | Use Case |
|--------------|-----|------------|----------|
| **Normal** | $f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | μ (mean), σ (std) | Natural phenomena |
| **Standard Normal** | Normal with μ=0, σ=1 | None | Z-scores |
| **Exponential** | $f(x) = \lambda e^{-\lambda x}$ | λ (rate) | Time between events |
| **Uniform** | $f(x) = \frac{1}{b-a}$ | a, b (bounds) | Equal probability |

**Normal Distribution (Gaussian):**

```
         ┌─────────────────────────────────────┐
         │           68.2%                      │
         │        ┌───────┐                     │
         │   95.4%│       │                     │
         │  ┌─────┴───────┴─────┐               │
         │  │     99.7%         │               │
    ─────┼──┴───────────────────┴──┬────────   │
        -3σ  -2σ  -1σ   μ   +1σ  +2σ  +3σ     │
         └─────────────────────────────────────┘
```

```python
# Normal distribution
normal = stats.norm(loc=100, scale=15)  # μ=100, σ=15

# Probabilities
print(f"P(X < 115): {normal.cdf(115):.4f}")
print(f"P(X > 85): {1 - normal.cdf(85):.4f}")
print(f"P(85 < X < 115): {normal.cdf(115) - normal.cdf(85):.4f}")

# Percentiles
print(f"95th percentile: {normal.ppf(0.95):.2f}")
```

**Z-Score (Standardization):**

$$Z = \frac{X - \mu}{\sigma}$$

```python
# Calculate Z-score
z_score = (value - mean) / std

# Using scipy
z = stats.zscore(df['column'])
```

**Confidence Intervals:**

$$CI = \bar{x} \pm z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}$$

```python
# 95% Confidence Interval
from scipy import stats
import numpy as np

data = df['column']
mean = np.mean(data)
se = stats.sem(data)  # Standard error
ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)
print(f"95% CI: {ci}")
```

---

## 3. Central Limit Theorem (CLT)

> **The most important theorem in statistics** - Regardless of population distribution, the sampling distribution of the mean approaches normal as sample size increases.

### 3.1 Key Concepts

| Concept | Description |
|---------|-------------|
| **Population** | Entire group of interest |
| **Sample** | Subset of population |
| **Sampling Distribution** | Distribution of sample statistics |
| **Sample Mean** | $\bar{x} = \frac{\sum x_i}{n}$ |
| **Standard Error** | $SE = \frac{\sigma}{\sqrt{n}}$ |

### 3.2 CLT Statement

For a population with mean μ and standard deviation σ:

$$\bar{X} \sim N\left(\mu, \frac{\sigma}{\sqrt{n}}\right) \text{ as } n \rightarrow \infty$$

**Rule of thumb:** n ≥ 30 for CLT to apply

```
Population (any distribution)
         │
         ▼
    Take many samples of size n
         │
         ▼
    Calculate mean of each sample
         │
         ▼
    Distribution of means → Normal!
```

### 3.3 Practical Application

```python
import numpy as np
import matplotlib.pyplot as plt

# Original population (exponential - not normal)
population = np.random.exponential(scale=2, size=100000)

# Sampling distribution of means
sample_means = []
for _ in range(1000):
    sample = np.random.choice(population, size=50, replace=True)
    sample_means.append(np.mean(sample))

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(population, bins=50, density=True)
axes[0].set_title('Population (Exponential)')
axes[1].hist(sample_means, bins=50, density=True)
axes[1].set_title('Sampling Distribution of Means (Normal!)')
plt.show()

# Verify CLT
print(f"Population mean: {np.mean(population):.2f}")
print(f"Mean of sample means: {np.mean(sample_means):.2f}")
print(f"SE (theoretical): {np.std(population)/np.sqrt(50):.4f}")
print(f"SE (empirical): {np.std(sample_means):.4f}")
```

### 3.4 Applications of CLT

| Application | How CLT Helps |
|-------------|---------------|
| **Confidence Intervals** | Construct intervals using normal distribution |
| **Hypothesis Testing** | Calculate p-values |
| **A/B Testing** | Compare sample means |
| **Quality Control** | Control charts |

---

## 4. Hypothesis Testing

### 4.1 Key Concepts

| Term | Definition |
|------|------------|
| **H₀ (Null Hypothesis)** | Default assumption (no effect/difference) |
| **H₁ (Alternative Hypothesis)** | What we want to prove |
| **α (Significance Level)** | Probability of Type I error (usually 0.05) |
| **p-value** | Probability of observing data if H₀ is true |
| **Test Statistic** | Calculated value to compare against distribution |

### 4.2 One-Tailed vs Two-Tailed Tests

| Test Type | H₁ | Use Case |
|-----------|-----|----------|
| **Two-tailed** | μ ≠ μ₀ | Detect any difference |
| **Left-tailed** | μ < μ₀ | Detect decrease |
| **Right-tailed** | μ > μ₀ | Detect increase |

```
Two-tailed (α = 0.05):          One-tailed (α = 0.05):
     ┌─────────────┐                 ┌─────────────┐
     │   Reject    │                 │             │
0.025│      │      │0.025           │             │0.05
─────┼──────┼──────┼─────        ───┼─────────────┼────
    -1.96   0    1.96                0           1.645
```

### 4.3 Critical Value vs P-value Approach

**Critical Value Approach:**
1. Set α (e.g., 0.05)
2. Find critical value from distribution
3. Calculate test statistic
4. Reject H₀ if test statistic > critical value

**P-value Approach:**
1. Calculate test statistic
2. Find p-value (probability of getting this extreme)
3. Reject H₀ if p-value < α

```python
# Both approaches give same conclusion
# P-value approach is more informative

alpha = 0.05
# If p-value < alpha → Reject H₀
```

### 4.4 Common Statistical Tests

| Test | Use Case | Assumptions |
|------|----------|-------------|
| **Z-test** | Large sample (n≥30), known σ | Normal distribution |
| **T-test** | Small sample, unknown σ | Normal distribution |
| **Chi-square** | Categorical variables | Expected freq ≥ 5 |
| **ANOVA** | Compare 3+ group means | Normal, equal variance |

**One-sample T-test:**
```python
from scipy import stats

# Test if mean differs from hypothesized value
sample_data = df['column']
hypothesized_mean = 100

t_stat, p_value = stats.ttest_1samp(sample_data, hypothesized_mean)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: Mean significantly differs from 100")
else:
    print("Fail to reject H₀: No significant difference")
```

**Two-sample T-test:**
```python
# Compare means of two groups
group_a = df[df['group'] == 'A']['value']
group_b = df[df['group'] == 'B']['value']

# Independent samples t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"P-value: {p_value:.4f}")
```

**Chi-square Test:**
```python
# Test independence of categorical variables
contingency_table = pd.crosstab(df['var1'], df['var2'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square: {chi2:.4f}, P-value: {p_value:.4f}")
```

### 4.5 A/B Testing

Comparing two versions (A = Control, B = Treatment).

```python
# A/B Test for conversion rates
from scipy import stats

# Data
conversions_a, total_a = 120, 1000  # Control
conversions_b, total_b = 145, 1000  # Treatment

# Conversion rates
rate_a = conversions_a / total_a
rate_b = conversions_b / total_b

# Pooled proportion
pooled = (conversions_a + conversions_b) / (total_a + total_b)

# Standard error
se = np.sqrt(pooled * (1 - pooled) * (1/total_a + 1/total_b))

# Z-statistic
z = (rate_b - rate_a) / se

# P-value (two-tailed)
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"Control: {rate_a:.2%}, Treatment: {rate_b:.2%}")
print(f"Lift: {(rate_b - rate_a) / rate_a:.2%}")
print(f"P-value: {p_value:.4f}")
```

### 4.6 Type I and Type II Errors

|  | H₀ True | H₀ False |
|--|---------|----------|
| **Reject H₀** | Type I Error (α) - False Positive | Correct (Power = 1-β) |
| **Fail to Reject H₀** | Correct | Type II Error (β) - False Negative |

| Error Type | Description | Example |
|------------|-------------|---------|
| **Type I (α)** | False positive - rejecting true H₀ | Convicting innocent person |
| **Type II (β)** | False negative - failing to reject false H₀ | Letting guilty go free |
| **Power (1-β)** | Probability of detecting true effect | Correctly finding difference |

```python
# Power analysis
from statsmodels.stats.power import TTestIndPower

power_analysis = TTestIndPower()

# Calculate required sample size
sample_size = power_analysis.solve_power(
    effect_size=0.5,  # Medium effect
    alpha=0.05,
    power=0.8,
    ratio=1  # Equal groups
)
print(f"Required sample size per group: {int(sample_size)}")
```

---

## 5. SQL & Data Management

### 5.1 Star Schema & Data Warehousing

```
                    ┌─────────────┐
                    │    Fact     │
                    │   Table     │
                    │  (Sales)    │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  ┌───────────┐     ┌───────────┐     ┌───────────┐
  │ Dimension │     │ Dimension │     │ Dimension │
  │  (Time)   │     │ (Product) │     │(Customer) │
  └───────────┘     └───────────┘     └───────────┘
```

| Component | Description | Example |
|-----------|-------------|---------|
| **Fact Table** | Measurements, metrics | Sales amount, quantity |
| **Dimension Table** | Descriptive attributes | Customer name, product category |
| **Foreign Keys** | Links fact to dimensions | customer_id, product_id |

### 5.2 Basic SQL

```sql
-- SELECT: Retrieve data
SELECT column1, column2 FROM table_name;

-- WHERE: Filter rows
SELECT * FROM sales WHERE amount > 100;

-- GROUP BY: Aggregate data
SELECT category, SUM(amount) as total_sales
FROM sales
GROUP BY category;

-- HAVING: Filter aggregated results
SELECT category, SUM(amount) as total_sales
FROM sales
GROUP BY category
HAVING SUM(amount) > 10000;

-- ORDER BY: Sort results
SELECT * FROM sales ORDER BY amount DESC;

-- JOIN: Combine tables
SELECT s.*, c.customer_name
FROM sales s
JOIN customers c ON s.customer_id = c.id;

-- Types of JOINs
-- INNER JOIN: Only matching rows
-- LEFT JOIN: All from left + matching from right
-- RIGHT JOIN: All from right + matching from left
-- FULL OUTER JOIN: All from both tables
```

### 5.3 Advanced SQL

**Window Functions:**
```sql
-- Running total
SELECT
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM sales;

-- Ranking
SELECT
    product_name,
    amount,
    RANK() OVER (ORDER BY amount DESC) as sales_rank,
    ROW_NUMBER() OVER (ORDER BY amount DESC) as row_num
FROM sales;

-- Partition by
SELECT
    category,
    product_name,
    amount,
    SUM(amount) OVER (PARTITION BY category) as category_total,
    amount / SUM(amount) OVER (PARTITION BY category) as pct_of_category
FROM sales;

-- LAG and LEAD
SELECT
    date,
    amount,
    LAG(amount, 1) OVER (ORDER BY date) as prev_amount,
    amount - LAG(amount, 1) OVER (ORDER BY date) as change
FROM sales;
```

**Indexing:**
```sql
-- Create index for faster queries
CREATE INDEX idx_customer_id ON sales(customer_id);

-- Composite index
CREATE INDEX idx_date_category ON sales(date, category);

-- When to use indexes:
-- - Frequently queried columns
-- - JOIN columns
-- - WHERE clause columns
-- - ORDER BY columns
```

**Common Table Expressions (CTE):**
```sql
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', date) as month,
        SUM(amount) as total
    FROM sales
    GROUP BY 1
),
monthly_growth AS (
    SELECT
        month,
        total,
        LAG(total) OVER (ORDER BY month) as prev_total,
        (total - LAG(total) OVER (ORDER BY month)) / LAG(total) OVER (ORDER BY month) as growth_rate
    FROM monthly_sales
)
SELECT * FROM monthly_growth;
```

---

## 6. Feature Engineering

> **80/20 Insight:** Feature Engineering creates 80% of the difference between a good model and an excellent model.

### 6.1 Encoding Techniques

| Technique | Description | Use Case |
|-----------|-------------|----------|
| **One-Hot** | Binary columns for each category | Low cardinality |
| **Label** | Integer mapping | Ordinal variables |
| **Target** | Replace with target mean | High cardinality |
| **Frequency** | Replace with frequency | High cardinality |

### 6.2 Scaling Techniques

| Technique | Formula | Use Case |
|-----------|---------|----------|
| **StandardScaler** | $z = \frac{x - \mu}{\sigma}$ | Linear models, NN |
| **MinMaxScaler** | $\frac{x - x_{min}}{x_{max} - x_{min}}$ | Neural Networks |
| **RobustScaler** | $\frac{x - median}{IQR}$ | Data with outliers |

### 6.3 Handling Imbalanced Data

| Technique | Type | Description |
|-----------|------|-------------|
| **SMOTE** | Oversampling | Synthetic minority samples |
| **ADASYN** | Oversampling | Adaptive synthetic |
| **Random Undersampling** | Undersampling | Remove majority |
| **Class Weights** | Algorithm | Penalize majority errors |

---

## 7. Model Evaluation Metrics

### 7.1 Classification Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN)/(Total) | Balanced classes |
| **Precision** | TP/(TP+FP) | FP cost is high |
| **Recall** | TP/(TP+FN) | FN cost is high |
| **F1-Score** | 2×(P×R)/(P+R) | Balance P & R |
| **AUC-ROC** | Area under ROC | Comparing models |

### 7.2 Regression Metrics

| Metric | Formula | Sensitivity to Outliers |
|--------|---------|-------------------------|
| **MAE** | $\frac{1}{n}\sum\|y - \hat{y}\|$ | Low |
| **MSE** | $\frac{1}{n}\sum(y - \hat{y})^2$ | High |
| **RMSE** | $\sqrt{MSE}$ | High |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Medium |

---

## 8. Common Pitfalls & Best Practices

### 8.1 Data Leakage

| Type | Description | Prevention |
|------|-------------|------------|
| **Target Leakage** | Feature contains target info | Domain knowledge review |
| **Train-Test Contamination** | Test influences training | Split before preprocessing |
| **Temporal Leakage** | Future info in past prediction | Time-based splits |

### 8.2 Best Practices Checklist

```
□ DATA UNDERSTANDING
  □ EDA completed
  □ Target distribution checked
  □ Missing values analyzed
  □ Outliers identified

□ DATA PREPARATION
  □ Split BEFORE preprocessing
  □ No data leakage
  □ Proper encoding
  □ Scaling applied

□ EVALUATION
  □ Correct metric chosen
  □ Cross-validation used
  □ Results make business sense
```

---

## 9. Assignment / Project

### NYC Taxi Dataset EDA

**Objectives:**
1. Data cleaning (missing values, outliers)
2. Analysis and visualization
3. Business insights and recommendations

**Project Structure:**
```
nyc_taxi_eda/
├── data/
│   └── nyc_taxi.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_insights.ipynb
└── reports/
    └── final_report.pdf
```

**Key Analysis Areas:**
- Trip duration analysis
- Pickup/dropoff patterns
- Fare prediction factors
- Time-based trends
- Geographic patterns

**Deliverables:**
- Clean dataset
- Visualization dashboard
- Business recommendations
- Presentation slides

---

## Key Takeaways

1. **EDA first** - Understand data before modeling
2. **CLT enables inference** - Sample means are normally distributed
3. **Choose the right test** - Match test to data type and hypothesis
4. **SQL is essential** - Data lives in databases
5. **Prevent data leakage** - Split before preprocessing

---

[← Back to Main Roadmap](../README.md) | [Next: Course 2 - ML Fundamentals →](../course2-ml-fundamentals/README.md)
