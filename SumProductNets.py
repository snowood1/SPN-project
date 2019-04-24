import numpy as np


class SumNode:
    def __init__(self, ch, w=None, name=None):
        self.ch = ch
        if w is None:
            self.w = np.ones(len(ch)) / len(ch)
        else:
            self.w = w
        self.p = list()
        self.scope = set()
        self.value = None
        self.name = name

    def __str__(self):
        if self.name is None:
            return self.__repr__()
        else:
            return self.name

    def eval(self):
        # return single value or a numpy array of values
        self.value = self.w @ [i.eval() for i in self.ch]
        return self.value

    def map_eval(self):
        # do not support batch operation
        self.value = self.w * [i.map_eval() for i in self.ch]
        return np.max(self.value)

    def map_back_track(self, output):
        # output is a dict
        i = self.ch[np.argmax(self.value)]
        i.map_back_track(output)


class ProductNode:
    def __init__(self, ch, name=None):
        self.ch = ch
        self.p = list()
        self.scope = set()
        self.value = None
        self.name = name

    def __str__(self):
        if self.name is None:
            return self.__repr__()
        else:
            return self.name

    def eval(self):
        # return single value or a numpy array of values
        self.value = 1
        for i in self.ch:
            self.value *= i.eval()
        return self.value

    def map_eval(self):
        self.value = 1
        for i in self.ch:
            self.value *= i.map_eval()
        return self.value

    def map_back_track(self, output):
        # output is a dict
        for i in self.ch:
            i.map_back_track(output)


class RVNode(SumNode):
    def __init__(self, rv, w=None):
        self.rv = rv

        ch = rv.leaf_nodes

        SumNode.__init__(self, ch, w)

        self.scope = {rv}

    def __str__(self):
        return self.rv.name


class LeafNode:
    def __init__(self, rv, domain_value):
        self.rv = rv
        self.domain_value = domain_value
        self.value = None

    def eval(self):
        return self.value

    def map_eval(self):
        return self.value

    def map_back_track(self, output):
        # output is a dict
        output[self.rv] = self.domain_value


class RV:
    def __init__(self, domain, name=None):
        self.domain = domain
        self.value = None
        self.leaf_nodes = list()
        for d in domain:
            self.leaf_nodes.append(LeafNode(self, d))
        self.name = name

    def __str__(self):
        if self.name is None:
            return self.__repr__()
        else:
            return self.name

    def set_value(self, value):
        self.value = value
        for i, d in zip(self.leaf_nodes, self.domain):
            if value is None:
                # when the value of rv is unknown, set all leaf node to one
                i.value = 1
            else:
                # when the value of rv is known, set the right leaf node to one, others to zero
                i.value = np.where(value == d, 1, 0)


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

    def print_weight(self):
        for n in self.nodes:
            if isinstance(n, SumNode):
                print(n, n.w)

    def init_scope(self, root):
        if type(root) is not RVNode:
            return root.scope
        else:
            scope = set()
            for i in root.ch:
                scope |= self.init_scope(i)
            root.scope = scope
            return scope

    def insert_data(self, obs_rvs, data):
        remaining_rvs = set(self.rvs)

        for i, rv in enumerate(obs_rvs):
            rv.set_value(data[i])
            remaining_rvs -= {rv}

        for rv in remaining_rvs:
            rv.set_value(None)

    def prob(self, obs_rvs, data):
        self.insert_data(obs_rvs, data)
        return self.root.eval()

    def map(self, obs_rvs, data):
        # do not support batch operation
        self.insert_data(obs_rvs, data)

        output = dict()
        self.root.map_eval()
        self.root.map_back_track(output)

        return np.array([output[rv] for rv in self.rvs])

    @staticmethod
    def softmax(x):
        res = np.e ** x
        return res / np.sum(res)

    def update_weight(self, data, step_size=1):
        s_g = {self.root: self.prob(self.rvs, data) ** -1}
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

    def init_tau(self):
        for n in self.nodes:
            if isinstance(n, SumNode):
                n.tau = np.random.rand(len(n.ch))
                n.w = self.softmax(n.tau)

    def train(self, data, iterations=100, step_size=1):
        self.init_tau()

        for itr in range(iterations):
            print('iteration:', itr)
            self.update_weight(data, step_size)
