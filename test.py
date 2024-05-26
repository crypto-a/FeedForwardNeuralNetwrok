from ffnn import ForwardFeed
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate the dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape y to be a 2D array for compatibility
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Initialize the neural network
ffnn = ForwardFeed(n_inputs=2)
ffnn.add_layer(n_neurons=4, activation='relu')
ffnn.add_layer(n_neurons=1, activation='softmax')
ffnn.compile()

# Train the neural network
ffnn.train(X_train, y_train, epochs=10000, learning_rate=0.1)

# Test the neural network
predictions = ffnn.predict(X_test)

# Evaluate the accuracy
predictions = (predictions > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)

print("Predicted Output:")
print(predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
