# Air Passengers Time Series Prediction

## Repository Overview

This repository contains two distinct approaches to predicting air passenger numbers using time series forecasting techniques:
1. Recurrent Neural Network (RNN) with LSTM
2. Custom ARIMA-like Model

### Dataset
- **Source**: Air Passengers Monthly Data
- **Description**: Monthly total number of international airline passengers from 1949 to 1960
- **Objective**: Predict future passenger numbers using different methodologies

## Project Structure

```
air-passengers-prediction/
│
├── data/
│   └── AirPassengers.csv
│
├── rnn_prediction.py
├── arima_prediction.py
│
├── README.md
└── requirements.txt
```

## Approaches

### 1. RNN (Recurrent Neural Network) Prediction
- **Model**: LSTM-based neural network
- **Key Features**:
  - Deep learning approach
  - Sequence-to-sequence prediction
  - Captures complex temporal dependencies
- **Preprocessing**:
  - Min-Max scaling
  - Sequence creation
  - 3D tensor transformation

### 2. Custom ARIMA-like Prediction
- **Model**: Linear Regression with RNN-inspired preprocessing
- **Key Features**:
  - Statistical time series approach
  - Custom sequence preparation
  - Flexible model parameters
- **Preprocessing**:
  - Similar to RNN approach
  - Linear regression-based prediction

## Dependencies

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow (for RNN)

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/air-passengers-prediction.git
cd air-passengers-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### RNN Prediction
```python
python rnn_prediction.py
```

### ARIMA-like Prediction
```python
python arima_prediction.py
```

## Performance Metrics

Both models are evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

## Visualization

Each implementation generates:
- Actual vs Predicted Passenger Numbers
- Loss Curves (for RNN)
- Prediction Error Analysis

## Comparative Analysis

### RNN Strengths
- Captures non-linear patterns
- Learns complex temporal dependencies
- Deep learning approach

### ARIMA-like Model Strengths
- Simpler implementation
- Statistical foundation
- Faster computation

## Limitations

- RNN:
  - Requires more data
  - Computationally expensive
  - Hyperparameter sensitive

- ARIMA-like Model:
