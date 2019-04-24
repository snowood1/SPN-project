import numpy as np
import pickle
import os

from SumProductNets import *
from random import shuffle, randint


# Network generator -- generate network with single parent
class NetGenerator(object):
    def __init__(self, num_feature, rv_list, sum_replicate=2, prod_replicate=2):
        '''
        Store the information that generator used to generate network
        :param num_feature: number of random variables of input data
        :param rv_list: predefined RV list, eg [rv1, rv2, rv3, ...]
        :param sum_replicate: The replicate factor at each sum node.
            Namely, create 'sum_replicate' prod_nodes.
            Accept both tuple and scalar -- tuple for random generate
        :param prod_replicate: The replicate factor at each prod node.
            Namely, create 'prod_replicate' sum_nodes if applicable.
            Accept both tuple and scalar -- tuple for random generate
        '''

        assert len(rv_list) == num_feature
        self.features = list(range(num_feature))
        self.rv_list = rv_list
        self.sum_rep = sum_replicate
        self.prod_rep = prod_replicate

    def generate(self):
        return self._create_node(self.features, True)

    def _create_node(self, var_list, sum_node=True):
        # Deal with base case, where the length of var_list equals one or zero
        if len(var_list) == 0:
            return None
        elif len(var_list) == 1:
            return RVNode(self.rv_list[var_list[0]])

        # Deal with general cases
        children = []
        if sum_node:
            # Generate Sum Node based on sum replicate factor
            if isinstance(self.sum_rep, tuple):
                current_replicate = randint(self.sum_rep[0], self.sum_rep[1])
            else:
                current_replicate = self.sum_rep
            for _ in range(current_replicate):
                tmp_node = self._create_node(var_list, False)
                if tmp_node is not None:
                    children.append(tmp_node)
            return SumNode(children)
        else:
            # Split the var_list into k different subsets, where k equals prod_rep
            shuffle(var_list)
            if isinstance(self.prod_rep, tuple):
                current_replicate = randint(self.prod_rep[0], self.prod_rep[1])
            else:
                current_replicate = self.prod_rep
            sub_size = len(var_list) // current_replicate
            
            sub_size = sub_size if sub_size != 0 else 1
            for c_id in range(0, len(var_list), sub_size):
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
    test_rv = [RV(domain=[0, 1]) for _ in range(788)]
    test_gen = NetGenerator(788, test_rv, sum_replicate=2, prod_replicate=4)
    root = test_gen.generate()
    test_spn = SPN(root, test_rv)
    # save2file(os.getcwd() + "/test.obj", test_spn)
    print(goThrough(test_spn.root))
    # test_spn = read_from_file(os.getcwd() + "/test.obj")
    # print(goThrough(test_spn.root))


if __name__ == "__main__":
    main()

