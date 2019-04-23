import numpy as np
import pickle
import os

from SumProductNets import *
from random import shuffle


# Network generator -- generate network with single parent
class NetGenerator(object):
    def __init__(self, num_feature, rv_list, sum_replicate=2, prod_replicate=2):
        self.features = list(range(num_feature))
        self.rv_list = rv_list
        self.sum_rep = sum_replicate
        self.prod_rep = prod_replicate

    def generate(self):
        return self._create_node(self.features, True)

    def _create_node(self, var_list, sum_node=True):
        # Deal with base case, where the length of var_list equals one & zero
        if len(var_list) == 0:
            return None
        elif len(var_list) == 1:
            return RVNode(self.rv_list[var_list[0]])

        # Deal with general cases
        children = []
        if sum_node:
            # Generate Sum Node based on sum replicate factor
            for _ in range(self.sum_rep):
                tmp_node = self._create_node(var_list, False)
                if tmp_node is not None:
                    children.append(tmp_node)
            return SumNode(children)
        else:
            # Split the var_list into k different subsets, where k equals prod_rep
            shuffle(var_list)
            sub_size = len(var_list) // self.prod_rep
            for c_id in range(0, len(var_list), sub_size if sub_size != 0 else 1):
                tmp_node = self._create_node(var_list[c_id:c_id + sub_size], True)
                if tmp_node is not None:
                    children.append(tmp_node)
            return ProductNode(children)


def save2file(path, root):
    with open(path, 'wb') as f:
        pickle.dump(root, f)


def read_from_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def goThrough(node):
    res = 1
    for ch in node.ch:
        if not isinstance(ch, LeafNode):
            res += goThrough(ch)
    return res


def main():
    test_rv = [RVNode(RV(domain=[0, 1])) for _ in range(2)]
    test_gen = NetGenerator(2, test_rv, sum_replicate=2, prod_replicate=2)
    root = test_gen.generate()
    test_spn = SPN(root, test_rv)
    # save2file(os.getcwd() + "/test.obj", test_spn)
    print(goThrough(test_spn.root))
    test_spn = read_from_file(os.getcwd() + "/test.obj")
    print(goThrough(test_spn.root))


if __name__ == "__main__":
    main()

