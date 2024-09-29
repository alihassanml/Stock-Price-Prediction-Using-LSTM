# Stock Price Prediction Using LSTM

[![License](https://img.shields.io/github/license/alihassanml/Stock-Price-Prediction-Using-LSTM)](LICENSE)
[![Python](https://img.shields.io/badge/python-v3.8%2B-blue)](https://www.python.org/downloads/release/python-380/)

This project aims to predict stock prices using Long Short-Term Memory (LSTM) neural networks. It provides a machine learning model to forecast Bitcoin prices based on historical data, utilizing TensorFlow for building and training the model and Streamlit for creating an interactive web app for predictions.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Streamlit App](#streamlit-app)
- [Files in Repository](#files-in-repository)
- [License](#license)

## Introduction

Predicting stock prices is a complex task, but LSTM (a type of recurrent neural network) can capture trends from time-series data. This project specifically focuses on Bitcoin price prediction using its historical data including features like `Open`, `High`, `Low`, and `Volume`.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/alihassanml/Stock-Price-Prediction-Using-LSTM.git
   cd Stock-Price-Prediction-Using-LSTM
   ```

2. Create a virtual environment and install the dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install the required Python libraries:

   - TensorFlow
   - Streamlit
   - Pandas
   - NumPy
   - Scikit-learn
   - Matplotlib
   - Plotly

   You can install them via:

   ```bash
   pip install tensorflow streamlit pandas numpy scikit-learn matplotlib plotly
   ```

## Usage

### 1. Training the Model

To train the LSTM model on the Bitcoin dataset, run:

```bash
python train_model.py
```

This will train the model and save it as `model.h5`.

### 2. Running the Streamlit App

After training the model, you can run the Streamlit app to interactively predict Bitcoin prices:

```bash
streamlit run app.py
```

In the Streamlit web interface, input the `Open`, `High`, `Low`, and `Volume` values for prediction, and the model will output the predicted closing price.

### 3. Predict Using Saved Model

If you have already trained the model and want to make predictions directly:

```python
import tensorflow as tf
import pickle
import numpy as np

# Load model and scalers
model = tf.keras.models.load_model('model.h5')
scalar_X = pickle.load(open('scalar_X.pkl', 'rb'))
scalar_y = pickle.load(open('scalar_y.pkl', 'rb'))

# Prepare the input data
data = np.array([[65180.66, 66480.0, 64852.99, 32058813449]])  # Example input

# Transform the data
scaled_data = scalar_X.transform(data)

# Make prediction
y_pred = model.predict(scaled_data)

# Inverse transform to get original scale
y_pred_original = scalar_y.inverse_transform(y_pred)

print('Predicted Close Price:', y_pred_original[0][0])
```

## Model

This LSTM model is designed to predict the Bitcoin price. It is trained on historical price data, and the input features used are `Open`, `High`, `Low`, and `Volume`. The target is the `Close` price. Min-Max scaling is used to normalize the data before feeding it into the model.

### Model Architecture:
- Input: 4 features (Open, High, Low, Volume)
- Hidden Layers: LSTM layers with dropout
- Output: Predicted Close price

## Streamlit App

The Streamlit app is designed to take user inputs (`Open`, `High`, `Low`, and `Volume`), run the LSTM model, and display the predicted close price. The app can be launched by running:

```bash
streamlit run app.py
```

### Screenshot

(Include a screenshot of your running Streamlit app here.)

## Files in Repository

- `app.py`: Streamlit app for predicting Bitcoin prices.
- `train_model.py`: Python script for training the LSTM model.
- `model.h5`: Pre-trained LSTM model for predicting stock prices.
- `scalar_X.pkl` and `scalar_y.pkl`: Pickled scalers for transforming input and output data.
- `BTC-Data.csv`: CSV file containing historical Bitcoin data for training.
- `requirements.txt`: Python dependencies for the project.
- `README.md`: Project documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
