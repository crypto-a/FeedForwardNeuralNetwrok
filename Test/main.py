from ffnn import ForwardFeed
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    ffnn = ForwardFeed(n_inputs=2)
    ffnn.add_layer(n_neurons=8, activation='relu')
    ffnn.add_layer(n_neurons=16, activation='relu')
    ffnn.add_layer(n_neurons=8, activation='relu')
    ffnn.add_layer(n_neurons=1, activation='sigmoid')
    ffnn.compile()

    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_moons(n_samples=10000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    ffnn.train(X_train, y_train, epochs=20000, learning_rate=0.01)

    predictions = ffnn.predict(X_test)
    predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(predictions == y_test)

    print("Predicted Output:")
    print(predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")