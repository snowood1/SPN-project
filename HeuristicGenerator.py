from SumProductNets import *
from random import shuffle, randint, random
import math

class HeuristicGenerator(object):
    def __init__(self, rv_list):
        self.rv_list = rv_list
        self.features = list(range(len(rv_list)))
        self.sum_rep = 2
        self.prod_rep = 2
        self.perline = 2

    def generate(self):
        return self._create_node(self.features, True)


    def _create_node(self, var_list, sum_node=True):
        # Deal with base case, where the length of var_list equals one or zero
        if len(var_list) == 0:
            return None

        elif len(var_list) == 1:
            # print(self.rv_list[var_list[0]])
            return RVNode(self.rv_list[var_list[0]])

        children = []
        if sum_node:
            # Generate Sum Node based on sum replicate factor
            if isinstance(self.sum_rep, tuple):
                current_replicate = randint(self.sum_rep[0], self.sum_rep[1])
            else:
                current_replicate = self.sum_rep
            for i in range(current_replicate):
                new_list = var_list * 1
                tmp_node = self._create_node(new_list, False)
                if tmp_node is not None:
                    children.append(tmp_node)
            return SumNode(children)
        else:
            if isinstance(self.prod_rep, tuple):
                current_replicate = randint(self.prod_rep[0], self.prod_rep[1])
            else:
                current_replicate = self.prod_rep
            sub_size = len(var_list) // current_replicate

            sub_size = sub_size if sub_size != 0 else 1

            for c_id in range(0, len(var_list), sub_size):
                # node_ =
                if var_list == []:
                    break
                tmp_node = []

                for i in range(sub_size):
                    if var_list == []:
                        break
                    index = self.add_node_index(tmp_node, var_list)
                    tmp_node.append(var_list[index])
                    var_list.pop(index)

                children.append(self._create_node(tmp_node, True))

            return ProductNode(children)

    def coordinate(self, val):
        if val >= self.perline ** 2:
            return -self.perline, -self.perline
        x = int(val % self.perline)
        y = int(val // self.perline)
        return x, y

    def center(self, list):
        total_x = 0
        total_y = 0
        for i in list:
            x, y = self.coordinate(i)
            total_x += x
            total_y += y
        return total_x / len(list), total_y / len(list)

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    def manhattan_distance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def weight_node(self, list1, node):
        x1, y1 = self.center(list1)
        x2, y2 = self.coordinate(node)
        return 1 / self.euclidean_distance(x1, y1, x2, y2)

    def conditoanl_probability(self, list1, list2):
        list3 = []
        normalizer = 0.0

        if list1 == []:
            for _ in list2:
                list3.append(1.0 / len(list2))
            return list3

        for node in list2:
            val = self.weight_node(list1, node)
            list3.append(val)
            normalizer += val

        for i in range(len(list3)):
            list3[i] = list3[i] / normalizer

        return list3

    def make_selection(self, prob_list):
        for i in range(1, len(prob_list)):
            prob_list[i] = prob_list[i - 1] + prob_list[i]
        r = random()
        # print(r)
        for i in range(len(prob_list)):
            if r <= prob_list[i]:
                return i
        return -1

    def add_node_index(self, list1, list2):
        # return randint(0, len(list2) - 1)
        prob_list = self.conditoanl_probability(list1, list2)
        return self.make_selection(prob_list)

    def convert_to_plain(self, x, y, num):
        return num * y + x

def goThrough(node):
    print(node.scope)
    for ch in node.ch:
        if not isinstance(ch, LeafNode):
            goThrough(ch)


def main():
    test_rv = [RV(domain=[0, 1]) for _ in range(4)]
    test_gen = HeuristicGenerator(test_rv)
    root = test_gen.generate()
    test_spn = SPN(root, test_rv)
    goThrough(test_spn.root)
    # save2file(os.getcwd() + "/test.json", test_spn)
    # print(goThrough(test_spn.root))
    # test_spn = read_from_file(os.getcwd() + "/test.json", SPN)
    # print(goThrough(test_spn.root))


if __name__ == "__main__":
    main()