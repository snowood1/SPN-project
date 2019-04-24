from SumProductNets import *
from NetGenerator import NetGenerator, goThrough, save2file
from Preprocessing import get_data_loader, generate_data
import numpy as np
import os

# each column is a data point
data = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

# create random variables
rv1 = RV((0, 1), 'x1')
rv2 = RV((0, 1), 'x2')

# create leaf nodes
X1 = RVNode(rv1)
X2 = RVNode(rv2)
X3 = RVNode(rv1)
X4 = RVNode(rv2)

rv_list = [rv1, rv2]
test_gen = NetGenerator(2, rv_list, sum_replicate=(2, 6), prod_replicate=(2, 6))

# build SPN tree
S_prod1 = ProductNode([X1, X2])
S_prod2 = ProductNode([X3, X4])

# root = SumNode([S_prod1, S_prod2])
root = test_gen.generate()

S = SPN(root, [rv1, rv2])
# print(goThrough(S.root))

# compute the join probability
print(S.prob([rv1, rv2], data))

# compute the marginal of rv1 (sum out rv2)
print(S.prob([rv1], data[[0], :]))

# alternative to data matrix, value can be directly assign to random variables
print(S.prob([rv1, rv2], [0, 1]))

S.train(data, iterations=1000, step_size=5)

S.print_weight()

print(S.prob([rv1, rv2], data))

# Demo code for MNIST data set

rv_list = [RV((0, 1), 'x' + str(i)) for i in range(788)]
test_gen = NetGenerator(788, rv_list, sum_replicate=2, prod_replicate=4)

root = test_gen.generate()
S = SPN(root, rv_list)

mnist_dl = get_data_loader()

for f, l in mnist_dl:
    inter = generate_data(f, l)
    S.train(inter, iterations=100, step_size=5)

# S.print_weight()
save2file(os.getcwd() + "/spn_with_weight.obj", S)

