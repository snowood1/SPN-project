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

    def eval(self):
        return self.rv.leaf_node_value[self.domain_idx]


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
            rv.set_value(data[i, :])
            remaining_rvs -= {rv}

        for rv in remaining_rvs:
            rv.set_value(None)

        return self.root.eval()
