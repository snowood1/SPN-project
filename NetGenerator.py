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
            assert isinstance(self.prod_rep, tuple)
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
                                             max_cut=np.random.randint(self.prod_rep[0], self.prod_rep[1] + 1))
                for sb in sub_blocks:
                    children.append(self._create_node_v2(sb, sum_node=True, coarse_mode=True))
                return ProductNode(children)
        else:
            # Handle the feature matrix in fine grain mode
            if e_r - s_r == 1 and e_c - s_c == 1:
                var_name = self.features_matrix[s_r, s_c]
                return RVNode(self.rv_list[var_name])

            children = list()
            if sum_node:
                for _ in range(self.sum_rep):
                    children.append(self._create_node_v2(var_block, sum_node=False, coarse_mode=False))
                return SumNode(children)
            else:
                sub_blocks = self._decompose(var_block, block_type='fine',
                                             max_cut=np.random.randint(self.prod_rep[0], self.prod_rep[1] + 1))

                for sb in sub_blocks:
                    children.append(self._create_node_v2(sb, sum_node=True, coarse_mode=False))
                return ProductNode(children)

    def _generic_structure(self, coarse_solution, fine_solution=1):
        # Generate the general network structure of SPN -- Complex version
        # Namely, every node can only have one parent
        # K = sum_replicate

        assert self.img_size % coarse_solution == 0
        # coarse_img_size = self.img_size // coarse_solution
        #
        # coarse_solution_map = dict()
        # fine_solution_map = dict()
        #
        # # Generate all possible coarse resolution img decomposition
        # for _w in range(1, coarse_img_size + 1):
        #     for _h in range(1, coarse_img_size + 1):
        #         # Generate trivial decomposition
        #         if _w == 1 and _h == 1:
        #             continue
        #
        #         # Generate non-trivial decomposition
        #         for _w_s in range(0, self.img_size - _w * coarse_solution + 1, coarse_solution):
        #             for _h_s in range(0, self.img_size - _h * coarse_solution + 1, coarse_solution):
        #                 # Name current block by top_left and bottom_right pixel (y_s, x_s, y_e, x_e)
        #                 tmp_key = (_h_s, _w_s, _h_s + _h * coarse_solution, _w_s + _w * coarse_solution)
        #                 coarse_solution_map[tmp_key] = self._coordinate2vr(tmp_key)
        #
        # if self.start_point_list is None:
        #     self._generate_start_points(coarse_solution)
        #
        # # Generate trivial decomposition -- That is, (coarse_resolution * coarse_resolution) block
        # for _w in range(1, coarse_solution + 1):
        #     for _h in range(1, coarse_solution + 1):
        #
        #         for _w_s in range(0, coarse_solution - _w + 1):
        #             for _h_s in range(0, coarse_solution - _h + 1):
        #                 for (sy, sx) in self.start_point_list:
        #                     tmp_key = (sy + _h_s, sx + _w_s, sy + _h_s + _h, sx + _w_s + _w)
        #                     fine_solution_map[tmp_key] = self._coordinate2vr(tmp_key)

        return self._create_node_v2((0, 0, self.img_size, self.img_size), sum_node=True, coarse_mode=True)

    def _coordinate2vr(self, coordinate):
        res = self.features_matrix[coordinate[0]:coordinate[2], coordinate[1]:coordinate[3]].flatten().tolist()
        return res

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
                target_block_pivot = np.random.randint(0, len(sub_blocks))
                target_block = sub_blocks[target_block_pivot]
                if (target_block[2] - target_block[0] <= 1) and (target_block[3] - target_block[1] <= 1):
                    continue

                cut_type = np.random.random()
                if (cut_type > .5 and target_block[3] - target_block[1] > 1) or \
                        (cut_type <= .5 and target_block[2] - target_block[0] <= 1):
                    # Cut vertically & there exists a legal cut line
                    cut_line = np.random.randint(target_block[1] + 1, target_block[3])

                    p1, p2 = (target_block[0], target_block[1], target_block[2], cut_line), \
                             (target_block[0], cut_line, target_block[2], target_block[3])

                    sub_blocks[target_block_pivot] = p1
                    sub_blocks.append(p2)
                else:
                    # Cut horizontally
                    if target_block[2] - target_block[0] <= 1:
                        continue

                    cut_line = np.random.randint(target_block[0] + 1, target_block[2])
                    p1, p2 = (target_block[0], target_block[1], cut_line, target_block[3]), \
                             (cut_line, target_block[1], target_block[2], target_block[3])

                    sub_blocks[target_block_pivot] = p1
                    sub_blocks.append(p2)
        elif block_type == 'coarse':
            for _ in range(max_cut):
                target_block_pivot = np.random.randint(0, len(sub_blocks))
                target_block = sub_blocks[target_block_pivot]

                if (target_block[2] - target_block[0] <= coarse_solution) and \
                        (target_block[3] - target_block[1] <= coarse_solution):
                    continue

                cut_type = np.random.random()
                if (cut_type > .5 and target_block[3] - target_block[1] > coarse_solution) or \
                        (cut_type <= .5 and target_block[2] - target_block[0] <= coarse_solution):
                    # Cut vertically & there exists a legal cut line

                    cut_seed = np.random.randint(1, (target_block[3] - target_block[1]) // coarse_solution)
                    cut_line = target_block[1] + coarse_solution * cut_seed

                    p1, p2 = (target_block[0], target_block[1], target_block[2], cut_line), \
                             (target_block[0], cut_line, target_block[2], target_block[3])

                    sub_blocks[target_block_pivot] = p1
                    sub_blocks.append(p2)
                else:
                    # Cut horizontally
                    if target_block[2] - target_block[0] <= coarse_solution:
                        continue

                    cut_seed = np.random.randint(1, (target_block[2] - target_block[0]) // coarse_solution)
                    cut_line = target_block[0] + coarse_solution * cut_seed

                    p1, p2 = (target_block[0], target_block[1], cut_line, target_block[3]), \
                             (cut_line, target_block[1], target_block[2], target_block[3])

                    sub_blocks[target_block_pivot] = p1
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
    test_gen = NetGenerator(28*28, 0, test_rv, sum_replicate=2, prod_replicate=(5, 10))
    # test_gen._generic_structure(4)
    # root = test_gen.generate()
    # test_spn = SPN(root, test_rv)
    # goThrough(test_spn.root)
    # save2file(os.getcwd() + "/test.json", test_spn)
    # print(goThrough(test_spn.root))
    # test_spn = read_from_file(os.getcwd() + "/test.json", SPN)
    # print(goThrough(test_spn.root))
    root = test_gen.generate(mode='generic')
    print(goThrough(root))
    root2 = test_gen.generate(mode='forest')
    print(goThrough(root2))


if __name__ == "__main__":
    main()

