import pickle

from SumProductNets import *
from random import shuffle, randint
import os
import json
import math
import random
import numpy as np


# Network generator -- generate network with single parent
class NetGenerator(object):
    def __init__(self, num_feature, number_label, rv_list, sum_replicate=2, prod_replicate=2, cut_limit=None):
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
        self.cut_limit = cut_limit

    def generate(self, mode='forest'):
        if mode == 'forest':
            return self._create_node(self.features, True)
        elif mode == 'generic':
            assert self.cut_limit is not None
            return self._generic_structure(4)

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

    def _create_node_v2(self, var_block, sum_node=True, coarse_mode=True, coarse_resolution=4):
        (s_r, s_c, e_r, e_c) = var_block
        if coarse_mode:  # Handle the feature matrix in coarse mode
            if e_r - s_r == coarse_resolution and e_c - s_c == coarse_resolution:
                return self._create_node_v2(var_block, sum_node=True, coarse_mode=False)

            # Non-trivial cases
            children = list()
            if sum_node:
                for _ in range(self.sum_rep):
                    children.append(self._create_node_v2(var_block, sum_node=False, coarse_mode=True))
                return SumNode(children)
            else:
                sub_blocks = self._decompose(var_block, block_type='coarse',
                                             max_cut=np.random.randint(self.cut_limit[0], self.cut_limit[1] + 1))
                for sb in sub_blocks:
                    children.append(self._create_node_v2(sb, sum_node=True, coarse_mode=True))
                return ProductNode(children)
        else:
            # Handle the feature matrix in fine grain mode
            if e_r - s_r == 1 and e_c - s_c == 1:
                return RVNode(self.rv_list[self.features_matrix[s_r, s_c]])

            children = list()
            if sum_node:
                for _ in range(self.sum_rep):
                    children.append(self._create_node_v2(var_block, sum_node=False, coarse_mode=False))
                return SumNode(children)
            else:
                sub_blocks = self._decompose(var_block, block_type='fine',
                                             max_cut=np.random.randint(self.cut_limit[0], self.cut_limit[1] + 1))

                for sb in sub_blocks:
                    children.append(self._create_node_v2(sb, sum_node=True, coarse_mode=False))
                return ProductNode(children)

    def _generic_structure(self, coarse_solution):
        assert self.img_size % coarse_solution == 0
        return self._create_node_v2((0, 0, self.img_size, self.img_size), sum_node=True, coarse_mode=True)

    def _coordinate2vr(self, coordinate):
        return self.features_matrix[coordinate[0]:coordinate[2], coordinate[1]:coordinate[3]].flatten().tolist()

    def _generate_start_points(self, coarse_solution):
        res = []

        for _w_s in range(0, self.img_size - coarse_solution + 1, coarse_solution):
            for _h_s in range(0, self.img_size - coarse_solution + 1, coarse_solution):
                res.append((_h_s, _w_s))

        self.start_point_list = res

    def _decompose(self, in_block, block_type='coarse', coarse_solution=4, max_cut=4):
        # block_type would be coarse & fine
        sub_blocks = [in_block]

        if block_type == 'fine':
            for _ in range(max_cut):
                target_block_idx = np.random.randint(0, len(sub_blocks))
                (s_r, s_c, e_r, e_c) = sub_blocks[target_block_idx]

                if (e_r - s_r == 1) and (e_c - s_c == 1):
                    # Reach elemental pixel, give up this cut
                    continue

                cut_type = np.random.random()
                if (cut_type > .5 and e_c - s_c > 1) or (cut_type <= .5 and e_r - s_r == 1):
                    # Cut vertically, exist a legal cut in column or no legal cut line in row
                    cut_line = np.random.randint(s_c + 1, e_c)
                    p1, p2 = (s_r, s_c, e_r, cut_line), (s_r, cut_line, e_r, e_c)
                else:
                    # Cut horizontally otherwise
                    if e_r - s_r == 1:
                        continue

                    cut_line = np.random.randint(s_r + 1, e_r)
                    p1, p2 = (s_r, s_c, cut_line, e_c), (cut_line, s_c, e_r, e_c)

                sub_blocks[target_block_idx] = p1
                sub_blocks.append(p2)
        elif block_type == 'coarse':
            for _ in range(max_cut):
                target_block_idx = np.random.randint(0, len(sub_blocks))
                (s_r, s_c, e_r, e_c) = sub_blocks[target_block_idx]

                if (e_r - s_r <= coarse_solution) and (e_c - s_c <= coarse_solution):
                    # Reach elemental pixel, give up this cut
                    continue

                cut_type = np.random.random()
                if (cut_type > .5 and e_c - s_c > coarse_solution) or (cut_type <= .5 and e_r - s_r <= coarse_solution):
                    # Cut vertically & there exists a legal cut line
                    cut_line = s_c + coarse_solution * np.random.randint(1, (e_c - s_c) // coarse_solution)
                    p1, p2 = (s_r, s_c, e_r, cut_line), (s_r, cut_line, e_r, e_c)
                else:
                    # Cut horizontally
                    if e_r - s_r <= coarse_solution:
                        continue

                    cut_line = s_r + coarse_solution * np.random.randint(1, (e_r - s_r) // coarse_solution)
                    p1, p2 = (s_r, s_c, cut_line, e_c), (cut_line, s_c, e_r, e_c)

                sub_blocks[target_block_idx] = p1
                sub_blocks.append(p2)
        else:
            raise Exception('Unknown Block_Type presented!!!')

        return sub_blocks


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
    res = 1
    for ch in node.ch:
        if not isinstance(ch, LeafNode):
            res += goThrough(ch)
    return res


def main():
    test_rv = [RV(domain=[0, 1]) for _ in range(28*28)]
    test_rv_2 = [RV(domain=[0, 1]) for _ in range(28*28)]
    # Generic construction requires sum_replicate & cut_limit
    test_gen = NetGenerator(28*28, 0, test_rv, sum_replicate=2, cut_limit=(5, 10))
    # Random construction requires sum_replicate & prod_replicate
    test_gen_2 = NetGenerator(28*28, 0, test_rv_2, prod_replicate=5, cut_limit=(1, 1))

    root = test_gen.generate(mode='generic')
    print(goThrough(root))
    root2 = test_gen_2.generate(mode='forest')
    print(goThrough(root2))


if __name__ == "__main__":
    main()

