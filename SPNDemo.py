from SumProductNets import *
import numpy as np


# each column is a data point
data = np.array([
    [1, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 0, 1, 1, 1]
])
d, m = data.shape

# create random variables
rv1 = RV((0, 1), 'x1')
rv2 = RV((0, 1), 'x2')

# create leaf nodes
X1 = RVNode(rv1)
X2 = RVNode(rv2)
X3 = RVNode(rv1)
X4 = RVNode(rv2)

rv_list = [rv1, rv2]

# build SPN tree
S_prod1 = ProductNode([X1, X2])
S_prod2 = ProductNode([X3, X4])

root = SumNode([S_prod1, S_prod2])

S = SPN(root, [rv1, rv2])

# compute the join probability
print(S.prob([rv1, rv2], data))

# compute the marginal of rv1 (sum out rv2)
print(S.prob([rv1], data[[0], :]))

# alternative to data matrix, value can be directly assign to random variables
print(S.prob([rv1, rv2], [0, 1]))

# training
# S.train(data, iterations=1000, step_size=5)

# batch training
S.init_weight()
for itr in range(100):
    batch_idx = np.random.choice(m, 3)  # batch size is 3
    batch = data[:, batch_idx]
    for batch_itr in range(10):
        S.update_weight(batch, step_size=0.01)

S.print_weight()
print(S.prob([rv1, rv2], data))

# MAP inference, do not support batch operation
print(S.map([], []))
print(S.map([rv1], [0]))
print(S.map([rv2], [1]))
