# Heart Risk Prediction Using Neural Networks

This repository contains a neural network model for predicting heart risk using a dataset of cardiovascular measurements. The model leverages Keras and Keras Tuner for building and optimizing the neural network.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Installation

To run this project, you will need to install the following libraries:

```bash
pip install pandas numpy matplotlib scikit-learn keras keras-tuner
```

## Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/heart-risk-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd heart-risk-prediction
    ```
3. Place the `cardio_dataset.csv` file in the project directory.
4. Run the notebook to preprocess data, build and evaluate the model.

## Data Preprocessing

The dataset is loaded and the features and target variable are separated. Data scaling is performed using MinMaxScaler.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dataset = pd.read_csv("cardio_dataset.csv")

# Split Target and Features
data = dataset.iloc[:, 0:7].values
target = dataset.iloc[:, 7].values
target = np.reshape(target, (-1, 1))

# Scaling
scaler_data = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler()

data_scaled = scaler_data.fit_transform(data)
target_scaled = scaler_target.fit_transform(target)
```

## Model Building

The neural network model is built using Keras. The architecture is optimized using Keras Tuner.

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(288, input_dim=7, activation="relu"))
model.add(Dropout(0.4))
# Add additional layers...
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])
model.summary()
```

## Hyperparameter Tuning

Keras Tuner is used for hyperparameter optimization.

```python
from keras_tuner.tuners import RandomSearch

def build_model(parameters):
    model = Sequential()
    # Add layers...
    model.compile(optimizer=parameters.Choice("optimizer", ["adam", "adadelta", "adagrad"]), 
                  loss=parameters.Choice("loss function", ["mse", "mae"]))
    return model

tuner = RandomSearch(build_model, objective="val_loss", max_trials=5, executions_per_trial=3, directory="project", project_name="Heart-Risk")

tuner.search(x_train, y_train, epochs=200, validation_data=(x_test, y_test))
```

## Model Evaluation

The model is trained and evaluated, and the results are plotted and printed.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Train the model
model.fit(x_train, y_train, epochs=1000, validation_split=0.2)

# Plot training history
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Evaluate the model
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 score:", r2)
```

## Results

The R2 score, actual, and predicted values are printed. The predictions are also inverse scaled for better interpretation.

```python
print("r2 score:", r2)
print("actual inverse scaled:", scaler_target.inverse_transform(y_test[:10].T))
print("predicted inverse scaled:", scaler_target.inverse_transform(y_pred[:10].T))
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---
