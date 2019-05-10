from SumProductNets import *
from StructureLearning import train, mutual_info
from utils import save_spn, load_spn
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt


X, _ = loadlocal_mnist(
    images_path='./Data/train-images.idx3-ubyte',
    labels_path='./Data/train-labels.idx1-ubyte'
)

X = X[:2000, :].T
d, m = X.shape

X[X >= 0.5] = 1
X[X < 0.5] = 0

# rvs = list()
# for i in range(d):
#     rvs.append(RV((0, 1)))
#
# spn = train(X[:, :100], rvs)
#
# save_spn('./Data/mnist_random_spn', spn)
spn, rvs = load_spn('./Data/mnist_random_spn')

a = X[:, [210, 200, 360, 479, 461, 474, 256, 214, 469, 600]]

a_ = spn.map(rvs[392:], a[392:])

plt.imshow(a_)
plt.show()


