from mlxtend.data import loadlocal_mnist


X, Y = loadlocal_mnist(
    images_path='./Data/train-images.idx3-ubyte',
    labels_path='./Data/train-labels.idx1-ubyte'
)

print(X.shape)
