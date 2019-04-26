import numpy as np
import random

from SumProductNets import *

class RandomGenerator(object):
    def __init__(self, num_feature, rv_list):
        self.features = list(range(num_feature))
        self.rv_list = rv_list

    def generate(self):
        return self._create_node(self.features, True)

    def node_selection(self, node_list):
        return random.randint(0, len(node_list)-1)

    def create_prod_node(self, node_list):
        children = []
        while node_list != []:
            temp_children = []
            num_nodes = random.randint(2, 5)
            for i in range(num_nodes):
                if node_list == []:
                    break
                s = self.node_selection(node_list)
                temp_children.append(node_list[s])
            
        return ProductNode(node_list)

    def create_sum_node(self, node_list):
        children = []
        num_nodes = random.randint(2,5)
        for i in range(num_nodes):
            children.append(ProductNode(node_list))
        return SumNode(children)

    def create_sub_tree(self, node_list):
        top_nodes = []
        while node_list != []:
            temp_node_list = []
            num_nodes = random.randint(8, 16)
            for i in range(num_nodes):
                if node_list == []:
                    break
                s = self.node_selection(node_list)
                temp_node_list.append(node_list[s])
                node_list.pop(s)
            top_nodes.append(self.create_sum_node())
