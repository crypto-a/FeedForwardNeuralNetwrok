from Network import FeedForward  # Ensure your neural network class is correctly imported
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
