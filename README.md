# FeedForward Neural Network

This project is my attempt to create a neural network from scratch, inspired by the videos of Professor Ali Ghodsi from a lecture at the University of Waterloo called STAT 940.

## My Vision

Throughout the summer of 2024, I delved deeply into the AI rabbit hole. My friends recommended that I build my own neural network library to enhance my understanding of the topic. As a result, I created this library.

## About

This project is a simple feedforward neural network that can be used for classification and regression tasks. I have coded different activation functions and loss functions that can be used in the network, allowing for customization. Furthermore, I plan to expand this library by adding different features as I continue my journey through learning AI.

## Example of Use

```python
Copy code
from Network import FeedForward
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Standardize the input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an instance of the FeedForward network
ffnn = FeedForward(n_inputs=X_train.shape[1])
ffnn.add_layer(layer_type='dense', n_neurons=6, activation='relu')
ffnn.add_layer(layer_type='dense', n_neurons=8, activation='relu')
ffnn.add_layer(layer_type='dense', n_neurons=3, activation='sigmoid')
ffnn.compile(loss='CCE', optimizer='ADAM')

# Train the network with a lower learning rate
ffnn.train(X_train, y_train, epochs=1000, learning_rate=0.001)

# Predict using the test set
y_pred_prob = ffnn.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_true = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```
## Output

```bash
Copy code
Epoch 0 - Loss: 0.6946227072444203
Epoch 100 - Loss: 0.7237607291898754
Epoch 200 - Loss: 0.48718487846538777
Epoch 300 - Loss: 0.24124330907480898
Epoch 400 - Loss: 0.08766527805945777
Epoch 500 - Loss: 0.06955778966107429
Epoch 600 - Loss: 0.06548118759959305
Epoch 700 - Loss: 0.06428484568768875
Epoch 800 - Loss: 0.06371188824312013
Epoch 900 - Loss: 0.06329536655383934
Accuracy: 100.00%
```

