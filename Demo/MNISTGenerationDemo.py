from SumProductNets import *
from StructureLearning import train, mutual_info
from utils import save_spn, load_spn
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show_images(images):
    gs = gridspec.GridSpec(3, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        img = img.reshape(28, 28)
        # img = np.transpose(img, [1, 2, 0])
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.show()


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

a = X[:, [210, 200, 360, 479, 461, 474, 256, 214, 469, 600]]

a_ = list()

for k in range(10):
    a_.append(a[:, k])

spn, rvs = load_spn('./Data/mnist_random_spn')
for k in range(10):
    a_.append(spn.map(rvs[392:], a[392:, k]))

spn, rvs = load_spn('./Data/mnist_learn_spn')
for k in range(10):
    a_.append(spn.map(rvs[392:], a[392:, k]))

show_images(a_)
