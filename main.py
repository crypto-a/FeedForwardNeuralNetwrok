from Layers import Dense


layer = Dense(2, 3, 'relu')

x = [[1, 2], [3, 4], [5, 6]]

print(layer.forward(x))