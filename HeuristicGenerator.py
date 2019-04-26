from SumProductNets import *
from random import shuffle, randint, random
import math

class HeuristicGenerator(object):
    def __init__(self, rv_list):
        self.rv_list = rv_list
        self.features = list(range(len(rv_list)))

    def generate(self):
        return self._create_node(self.features, True)


    def _create_node(self, var_list, sum_node=True):
        # Deal with base case, where the length of var_list equals one or zero
        if len(var_list) == 0:
            return None

        elif len(var_list) == 1:
            return RVNode(self.rv_list[var_list[0]])

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
            # shuffle(var_list)
            if isinstance(self.prod_rep, tuple):
                current_replicate = randint(self.prod_rep[0], self.prod_rep[1])
            else:
                current_replicate = self.prod_rep
            sub_size = len(var_list) // current_replicate

            sub_size = sub_size if sub_size != 0 else 1
            # for c_id in range(0, len(var_list), sub_size):
            #     tmp_node = self._create_node(var_list[c_id:c_id + sub_size], True)
            #     if tmp_node is not None:
            #         children.append(tmp_node)
            for c_id in range(0, len(var_list), sub_size):
                tmp_node = []
                for i in range(sub_size):


            return ProductNode(children)

    def coordinate(self, val, num):
        if val >= num ** 2:
            return -num, -num
        x = int (val % num)
        y = int(val // num)
        return x, y

    def center(self, list):
        total_x = 0
        total_y = 0
        for i in list:
            x, y = self.coordinate(i)
            total_x += x
            total_y += y
        return x / len(list), y / len(list)

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    def manhattan_distance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def weight_node(self, list, node):
        x1, y1 = self.center(list)
        x2, y2 = self.coordinate(node)
        return 1 / self.euclidean_distance(x1, y1, x2, y2)

    def conditoanl_probability(self, list1, list2):
        list3 = []
        normalizer = 0.0

        for node in list2:
            val = self.weight_node(list, node)
            list3.append(val)
            normalizer += val

        for i in range(list3):
            list3[i] = list3[i] / normalizer

        return list3

    def make_selection(self, prob_list):
        for i in range(1, len(prob_list)):
            prob_list[i] = prob_list[i-1] + prob_list[i]
        r = random()
        for i in range(len(prob_list)):
            if r <= prob_list[i]:
                return i
        return -1