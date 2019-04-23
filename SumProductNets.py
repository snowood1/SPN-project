import numpy as np
import itertools


class SumNode:
    def __init__(self, ch, w=None):
        self.ch = ch
        if w is None:
            self.w = np.ones(len(ch)) / len(ch)
        else:
            self.w = w
        self.p = list()
        self.scope = set()
        self.value = None

    def eval(self):
        # return single value or a numpy array of values
        self.value = self.w @ [i.eval() for i in self.ch]
        return self.value


class ProductNode:
    def __init__(self, ch):
        self.ch = ch
        self.p = list()
        self.scope = set()
        self.value = None

    def eval(self):
        # return single value or a numpy array of values
        self.value = 1
        for i in self.ch:
            self.value *= i.eval()
        return self.value


class LeafNode:
    def __init__(self, rv, domain_idx):
        self.rv = rv
        self.domain_idx = domain_idx
        self.value = None

    def eval(self):
        self.value = self.rv.leaf_node_value[self.domain_idx]
        return self.value


class RVNode(SumNode):
    def __init__(self, rv, w=None):
        self.rv = rv

        ch = list()
        for idx in range(len(rv.domain)):
            ch.append(LeafNode(rv, idx))

        SumNode.__init__(self, ch, w)

        self.scope = {rv}


class RV:
    id_counter = itertools.count()

    def __init__(self, domain, name=None):
        self.domain = domain
        self.value = None
        self.leaf_node_value = None
        if name is None:
            self.name = 'RV#' + next(self.id_counter)
        else:
            self.name = name

    def __str__(self):
        return self.name

    def set_value(self, value):
        self.value = value
        self.leaf_node_value = list()
        for d in self.domain:
            if value is None:
                # when the value of rv is unknown, set all leaf node to one
                leaf_value = 1
            else:
                # when the value of rv is known, set the right leaf node to one, others to zero
                leaf_value = np.where(value == d, 1, 0)

            self.leaf_node_value.append(leaf_value)


class SPN:
    def __init__(self, root, rvs):
        self.root = root
        self.rvs = rvs
        self.init_scope(root)

        self.nodes = list()  # a top down list of nodes

        queue = [root]
        while len(queue) > 0:
            n = queue.pop(0)
            self.nodes.append(n)
            if type(n) is not LeafNode:
                queue.extend(n.ch)

    def init_scope(self, root):
        if type(root) is not RVNode:
            return root.scope
        else:
            scope = set()
            for i in root.ch:
                scope |= self.init_scope(i)
            root.scope = scope
            return scope

    def prod(self, obs_rvs, data):
        remaining_rvs = set(self.rvs)

        for i, rv in enumerate(obs_rvs):
            rv.set_value(data[i])
            remaining_rvs -= {rv}

        for rv in remaining_rvs:
            rv.set_value(None)

        return self.root.eval()

    @staticmethod
    def softmax(x):
        res = np.e ** x
        return res / np.sum(res)

    def update_weight(self, data, step_size=1):
        print(self.prod(self.rvs, data))

        s_g = {self.root: 1}
        for n in self.nodes:
            if isinstance(n, SumNode):
                w_g = np.zeros(len(n.ch))
                for idx, j in enumerate(n.ch):
                    s_g[j] = s_g.get(j, 0) + n.w[idx] * s_g[n]
                    w_g[idx] = np.average(s_g[n] * j.value)
                tau_g = n.w * (w_g - np.sum(w_g * n.w))
                n.tau += tau_g * step_size
                n.w = self.softmax(n.tau)

            elif isinstance(n, ProductNode):
                for j in n.ch:
                    temp = 1
                    for k in n.ch:
                        if k is not j:
                            temp *= k.value
                    s_g[j] = s_g.get(j, 0) + s_g[n] * temp

    def train(self, data, iterations=100, step_size=1):
        for n in self.nodes:
            if isinstance(n, SumNode):
                n.tau = np.zeros(len(n.ch))

        for itr in range(iterations):
            print('iteration:', itr)
            self.update_weight(data, step_size)
