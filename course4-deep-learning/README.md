# Course 4: Deep Learning

> Master neural networks for complex pattern recognition in images, text, and sequential data.

![Course 4 Roadmap](./Course%202-3-4.PNG)

---

## Table of Contents

1. [Neural Networks Fundamentals](#1-neural-networks-fundamentals)
2. [Convolutional Neural Networks (CNN)](#2-convolutional-neural-networks-cnn)
3. [Recurrent Neural Networks (RNN)](#3-recurrent-neural-networks-rnn)
4. [Transfer Learning](#4-transfer-learning)
5. [Advanced Topics](#5-advanced-topics)
6. [When to Use Deep Learning](#6-when-to-use-deep-learning)

---

## 1. Neural Networks Fundamentals

Multi-layer perceptrons for learning complex non-linear patterns.

### 1.1 Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
     ↓              ↓               ↓
  Features    Learned patterns   Prediction
```

**Mathematical Formulation:**

For a single neuron:
$$z = \sum_{i=1}^{n} w_i x_i + b$$
$$a = \sigma(z)$$

Where:
- $x_i$ = input features
- $w_i$ = weights
- $b$ = bias
- $\sigma$ = activation function

### 1.2 Industry Applications

| Industry | Use Case |
|----------|----------|
| HR Analytics | Predicting employee turnover/attrition |
| Finance | Complex risk modeling |
| Manufacturing | Quality prediction |
| Image Classification | Simple image recognition |

### 1.3 Activation Functions

| Function | Formula | Output Range | Use Case |
|----------|---------|--------------|----------|
| **ReLU** | $\max(0, x)$ | [0, ∞) | Hidden layers (default) |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | (0, 1) | Binary classification output |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | Hidden layers |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | (0, 1), sum=1 | Multi-class output |
| **Leaky ReLU** | $\max(0.01x, x)$ | (-∞, ∞) | Avoiding dead neurons |

### 1.4 Loss Functions

| Task | Loss Function | Formula |
|------|---------------|---------|
| Regression | MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ |
| Regression | MAE | $\frac{1}{n}\sum|y - \hat{y}|$ |
| Binary Classification | Binary Cross-Entropy | $-\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$ |
| Multi-class | Categorical Cross-Entropy | $-\sum y_i \log(\hat{y}_i)$ |

### 1.5 Optimizers

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| **SGD** | Basic gradient descent | Simple problems |
| **Adam** | Adaptive learning rate | Default choice |
| **RMSprop** | Root mean square propagation | RNNs |
| **AdamW** | Adam with weight decay | Large models |

### 1.6 Regularization Techniques

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| **Dropout** | Randomly drop neurons | Overfitting |
| **L2 (Weight Decay)** | Penalize large weights | Always good practice |
| **Early Stopping** | Stop when val loss increases | Default practice |
| **Batch Normalization** | Normalize layer inputs | Deeper networks |
| **Data Augmentation** | Artificially increase data | Image tasks |

### 1.7 Implementation with Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train with early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)
```

### 1.8 Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **Learning Rate** | Step size (MOST IMPORTANT) | 1e-4 to 1e-2 |
| **Batch Size** | Samples per update | 16, 32, 64, 128 |
| **Epochs** | Training iterations | Use early stopping |
| **Hidden Layers** | Network depth | 2-5 for tabular |
| **Neurons per Layer** | Network width | 32-512 |
| **Dropout Rate** | Regularization strength | 0.2-0.5 |

---

## 2. Convolutional Neural Networks (CNN)

Specialized for processing grid-like data (images).

### 2.1 Industry Applications

| Industry | Use Case |
|----------|----------|
| Security | Face detection and recognition |
| Healthcare | Medical image analysis (CT, X-ray) |
| Automotive | Object detection for autonomous vehicles |
| Agriculture | Crop disease detection |

### 2.2 Architecture

```
Input Image → [Conv → ReLU → Pool] × N → Flatten → Dense → Output
```

**Key Components:**

| Layer | Purpose | Output |
|-------|---------|--------|
| **Convolution** | Feature detection | Feature maps |
| **Pooling** | Downsampling | Reduced size |
| **Flatten** | 2D to 1D | Vector |
| **Dense** | Classification | Predictions |

### 2.3 Convolution Operation

```
Input (5x5)         Kernel (3x3)        Output (3x3)
┌─────────────┐     ┌─────────┐         ┌───────┐
│ 1 2 3 4 5   │     │ 1 0 1   │         │ 14 .. │
│ 6 7 8 9 10  │  *  │ 0 1 0   │    =    │ .. .. │
│ 11 12 13 .. │     │ 1 0 1   │         │ .. .. │
│ ...         │     └─────────┘         └───────┘
└─────────────┘
```

**Parameters:**
- **Filters:** Number of feature detectors
- **Kernel Size:** Size of sliding window (3x3, 5x5)
- **Stride:** Step size of sliding
- **Padding:** 'same' (preserve size) or 'valid' (no padding)

### 2.4 Pooling

| Type | Operation | Purpose |
|------|-----------|---------|
| **Max Pooling** | Take maximum value | Most common |
| **Average Pooling** | Take average value | Smoother features |
| **Global Average Pooling** | Average entire feature map | Before dense layer |

### 2.5 Implementation

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    # Convolutional blocks
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 2.6 Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Train with augmentation
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_val, y_val)
)
```

### 2.7 Famous Architectures

| Model | Year | Parameters | Key Innovation |
|-------|------|------------|----------------|
| **LeNet** | 1998 | 60K | First successful CNN |
| **AlexNet** | 2012 | 60M | ReLU, Dropout |
| **VGG** | 2014 | 138M | Small kernels (3x3) |
| **ResNet** | 2015 | 25M | Skip connections |
| **EfficientNet** | 2019 | 5-66M | Compound scaling |
| **Vision Transformer** | 2020 | 86M+ | Self-attention |

---

## 3. Recurrent Neural Networks (RNN)

Designed for sequential/time-series data.

### 3.1 Industry Applications

| Industry | Use Case |
|----------|----------|
| Finance | Stock price prediction from time series |
| NLP | Sentiment analysis from product reviews |
| IoT | Sensor data prediction |
| Healthcare | Patient monitoring |

### 3.2 Architecture Evolution

```
Vanilla RNN → LSTM → GRU → Transformer
     ↓          ↓       ↓         ↓
  Vanishing   Long     Simpler   Attention
  gradient    memory   LSTM      mechanism
```

### 3.3 Vanilla RNN

```
    ┌─────┐     ┌─────┐     ┌─────┐
x_1 │     │ x_2 │     │ x_3 │     │
 ─► │ RNN │ ──► │ RNN │ ──► │ RNN │ ──► Output
    │     │     │     │     │     │
    └─────┘     └─────┘     └─────┘
       │           │           │
       ▼           ▼           ▼
      h_1         h_2         h_3 (hidden states)
```

**Problem:** Vanishing gradients for long sequences

### 3.4 LSTM (Long Short-Term Memory)

**Key Innovation:** Memory cell with gates

| Gate | Purpose |
|------|---------|
| **Forget Gate** | What to forget from memory |
| **Input Gate** | What new info to add |
| **Output Gate** | What to output |

```python
from tensorflow.keras.layers import LSTM

model = keras.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(1)
])
```

### 3.5 GRU (Gated Recurrent Unit)

Simpler than LSTM, often similar performance.

```python
from tensorflow.keras.layers import GRU

model = keras.Sequential([
    layers.GRU(128, return_sequences=True, input_shape=(timesteps, features)),
    layers.GRU(64),
    layers.Dense(1)
])
```

### 3.6 Bidirectional RNN

Process sequence in both directions.

```python
from tensorflow.keras.layers import Bidirectional

model = keras.Sequential([
    layers.Bidirectional(
        layers.LSTM(64, return_sequences=True),
        input_shape=(timesteps, features)
    ),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1)
])
```

### 3.7 Model Comparison

| Model | Memory | Speed | Best For |
|-------|--------|-------|----------|
| **LSTM** | Long-term | Slower | Long sequences, complex patterns |
| **GRU** | Long-term | Faster | Shorter sequences |
| **1D CNN** | Local patterns | Fast | Fixed patterns in sequences |
| **Transformer** | Very long | Parallel | Very long sequences, NLP |

### 3.8 Time Series Example

```python
import numpy as np

# Create sequences for time series
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Prepare data
seq_length = 60  # 60 time steps
X, y = create_sequences(scaled_data, seq_length)

# Build model
model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    layers.LSTM(50),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

---

## 4. Transfer Learning

Use pre-trained models as starting point.

### 4.1 Strategy by Data Size

| Data Size | Approach |
|-----------|----------|
| < 1k images | Feature extraction only |
| 1k - 10k | Fine-tune top layers |
| > 10k | Fine-tune entire network |

### 4.2 Pre-trained Models

| Model | Parameters | Accuracy | Speed |
|-------|------------|----------|-------|
| **ResNet50** | 25M | High | Medium |
| **EfficientNetB0** | 5M | High | Fast |
| **MobileNetV2** | 3M | Good | Very Fast |
| **VGG16** | 138M | Good | Slow |

### 4.3 Implementation

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Load pre-trained model (without top layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train top layers
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Fine-tune (unfreeze some layers)
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

---

## 5. Advanced Topics

### 5.1 Transformers

The dominant architecture for NLP and increasingly for vision.

**Key Innovation:** Self-attention mechanism

```python
# Using Hugging Face Transformers
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')

# Predict
outputs = model(inputs)
```

### 5.2 Generative Models

| Type | Purpose | Examples |
|------|---------|----------|
| **VAE** | Variational Autoencoder | Image generation |
| **GAN** | Generative Adversarial Network | Realistic images |
| **Diffusion** | Noise-based generation | DALL-E, Stable Diffusion |

### 5.3 Model Interpretability

```python
# Grad-CAM for CNN visualization
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()
```

---

## 6. When to Use Deep Learning

### 6.1 DL vs Traditional ML

| Use Deep Learning | Use Traditional ML |
|-------------------|-------------------|
| Large datasets (>100k) | Small datasets |
| Unstructured data (images, text, audio) | Tabular data |
| Complex patterns | Interpretability needed |
| Compute resources available | Limited resources |
| State-of-the-art needed | Good enough solution works |

### 6.2 Decision Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    CHOOSE YOUR APPROACH                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data Type?                                                  │
│       │                                                      │
│       ├── Tabular ────────────► XGBoost/LightGBM            │
│       │                                                      │
│       ├── Images ─────────────► CNN + Transfer Learning     │
│       │                                                      │
│       ├── Text ───────────────► Transformers (BERT)         │
│       │                                                      │
│       └── Time Series ────────► LSTM/GRU or XGBoost         │
│                                                              │
│  Data Size < 10k?                                           │
│       │                                                      │
│       ├── Yes ────────────────► Traditional ML + Regularize │
│       │                                                      │
│       └── No ─────────────────► Deep Learning viable        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Practice Projects

| Model | Project | Dataset |
|-------|---------|---------|
| Neural Network | Employee attrition | IBM HR |
| CNN | Image classification | CIFAR-10 |
| CNN + Transfer | Medical image | ChestX-ray |
| RNN/LSTM | Stock prediction | Yahoo Finance |
| LSTM | Sentiment analysis | IMDB Reviews |

---

## Key Takeaways

1. **Start simple** - Don't jump to deep learning if simpler models work
2. **Transfer learning first** - Use pre-trained models when possible
3. **Data quality > Model complexity** - More data often beats better architecture
4. **Regularize always** - Dropout, early stopping, data augmentation
5. **Learning rate matters most** - Tune this first

---

[← Back to Course 3](../course3-ml-advanced/README.md) | [Main Roadmap](../README.md)
