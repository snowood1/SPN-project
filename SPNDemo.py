from SumProductNets import *
import numpy as np

# each column is a data point
data = np.array([
    [0, 1, 1],
    [1, 1, 0]
])

# create random variables
rv1 = RV((0, 1), 'x1')
rv2 = RV((0, 1), 'x2')

# create leaf nodes
X1 = RVNode(rv1)
X2 = RVNode(rv2)
X3 = RVNode(rv1)
X4 = RVNode(rv2)

# build SPN tree
S_prod1 = ProductNode([X1, X2])
S_prod2 = ProductNode([X3, X4])

root = SumNode([S_prod1, S_prod2])

S = SPN(root, [rv1, rv2])

# compute the join probability
print(S.prod([rv1, rv2], data))

# compute the marginal or rv1 (sum out rv2)
print(S.prod([rv1], data[[0], :]))
