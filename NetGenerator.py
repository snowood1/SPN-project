import pickle

from SumProductNets import *
from random import shuffle, randint
import os
import json
import math
import numpy as np


# Network generator -- generate network with single parent
class NetGenerator(object):
    def __init__(self, num_feature, number_label, rv_list, sum_replicate=2, prod_replicate=2):
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

        assert len(rv_list) == (num_feature + number_label)
        self.img_size = int(math.sqrt(num_feature))
        self.features = list(range(num_feature + number_label))  # Flatten feature representation
        self.features_matrix = np.arange(num_feature).reshape(self.img_size, self.img_size)
        self.rv_list = rv_list
        self.sum_rep = sum_replicate
        self.prod_rep = prod_replicate
        self.start_point_list = None

    def generate(self, mode='forest'):
        if mode == 'forest':
            return self._create_node(self.features, True)
        elif mode == 'generic':
            return self._generate_start_points(4)

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

    def _generic_structure(self, coarse_solution, fine_solution=1):
        # Generate the general network structure of SPN -- Complex version
        # Namely, every node can only have one parent
        # K = sum_replicate

        assert self.img_size % coarse_solution == 0
        coarse_img_size = self.img_size // coarse_solution

        coarse_solution_map = dict()
        fine_solution_map = dict()

        # Generate all possible coarse resolution img decomposition
        for _w in range(1, coarse_img_size + 1):
            for _h in range(1, coarse_img_size + 1):
                # Generate trivial decomposition
                if _w == 1 and _h == 1:
                    continue

                # Generate non-trivial decomposition
                for _w_s in range(0, self.img_size - _w * coarse_solution + 1, coarse_solution):
                    for _h_s in range(0, self.img_size - _h * coarse_solution + 1, coarse_solution):
                        # Name current block by top_left and bottom_right pixel (y_s, x_s, y_e, x_e)
                        tmp_key = (_h_s, _w_s, _h_s + _h * coarse_solution, _w_s + _w * coarse_solution)
                        coarse_solution_map[tmp_key] = self._get_vr_by_coordinate(tmp_key)

        if self.start_point_list is None:
            self._generate_start_points(coarse_solution)

        # Generate trivial decomposition -- That is, (coarse_resolution * coarse_resolution) block
        for _w in range(1, coarse_solution + 1):
            for _h in range(1, coarse_solution + 1):

                for _w_s in range(0, coarse_solution - _w + 1):
                    for _h_s in range(0, coarse_solution - _h + 1):
                        for (sy, sx) in self.start_point_list:
                            tmp_key = (sy + _h_s, sx + _w_s, sy + _h_s + _h, sx + _w_s + _w)
                            fine_solution_map[tmp_key] = self._get_vr_by_coordinate(tmp_key)
        pass

    def _get_vr_by_coordinate(self, coordinate):
        res = self.features_matrix[coordinate[0]:coordinate[2], coordinate[1]:coordinate[3]].flatten().tolist()
        return res

    def _generate_start_points(self, coarse_solution):
        res = []

        for _w_s in range(0, self.img_size - coarse_solution + 1, coarse_solution):
            for _h_s in range(0, self.img_size - coarse_solution + 1, coarse_solution):
                res.append((_h_s, _w_s))

        self.start_point_list = res



def save2file(path, obj):
    with open(path, 'wb') as f:
        json.dump(obj.__dict__, f)


def read_from_file(path, cls):
    with open(path, 'rb') as f:
        obj_setting = json.load(f)
        instance = object.__new__(cls)

        for k, v in obj_setting:
            setattr(instance, k, v)

        return instance


def goThrough(node):
    print(node.convert2json())
    for ch in node.ch:
        if not isinstance(ch, LeafNode):
            goThrough(ch)


def main():
    test_rv = [RV(domain=[0, 1]) for _ in range(144)]
    test_gen = NetGenerator(144, 0, test_rv, sum_replicate=2, prod_replicate=2)
    test_gen._generic_structure(4)
    # root = test_gen.generate()
    # test_spn = SPN(root, test_rv)
    # goThrough(test_spn.root)
    # save2file(os.getcwd() + "/test.json", test_spn)
    # print(goThrough(test_spn.root))
    # test_spn = read_from_file(os.getcwd() + "/test.json", SPN)
    # print(goThrough(test_spn.root))


if __name__ == "__main__":
    main()

