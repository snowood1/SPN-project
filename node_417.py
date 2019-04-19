# Consider Leaf Node as Variables  SumNode

import numpy as np


class Node(object):
    def __init__(self, id, children=[],parents=[]):
        self.id = id
        self.parents = parents
        self.children = children
        self.scope= set()
        self.update_links()

    def update_links(self):
        for child in self.children:
            self.scope = self.scope.union(child.scope)
            if self not in child.parents:
                child.add_parent(self)

    def add_child(self,node):
        pass

    def add_parent(self,parent):
        self.parents.append(parent)

    def eval(self):
        pass

class SumNode(Node):
    def __init__(self, id, children=[],weights=[],parents=[]):

        assert all(isinstance(x, ProdNode) for x in children), 'Wrong child nodes'
        assert all(x.scope == children[0].scope for x in children), 'Children with an identical scope'
        assert len(weights)==len(children) or weights==[], 'Wrong dimension of weights'

        Node.__init__(self, id, children,parents)
        self.weights = weights if weights else np.ones(len(self.children))


    def add_child(self,node,weight=1):
        assert isinstance(node, ProdNode), 'Wrong child nodes'
        self.children.append(node)
        node.add_parent(self)
        self.weights.append(weight)

        # Todo:  MPE Inference (MPE assignment)

    def eval(self, X, mpe=None):  #  X = {x1:0, x2:0 , x3:1}
        s = 0.0
        weights = self.weights/np.sum(self.weights)
        if mpe == None:
            for child, weight in zip(self.children, weights):
                s += weight *child.eval(X)
        return s

class ProdNode(Node):
    def __init__(self, id, children=[],parents=[]):

        assert all(isinstance(x, (SumNode,LeafNode)) for x in children), 'Wrong child nodes'
        assert self.all_disjoint(children), 'Children with disjoint scopes'

        Node.__init__(self, id, children,parents)

    def all_disjoint(self,children):
        sets = [x.scope for x in children]
        union = set().union(*sets)
        n = sum(len(u) for u in sets)
        return n == len(union)

    def add_child(self,node):
        assert isinstance(node,(SumNode,LeafNode)), 'Wrong child nodes'
        node.add_parent(self)

    def eval(self, X):  #  X = {x1:0, x2:0 , x3:1}
        p = 1.0
        for child in self.children:
            p *= child.eval(X)
        return p


class LeafNode(Node):
    def __init__(self, id, variable, domain=[0, 1], weights=[0.5, 0.5], parents=[]):

        '''
        One leaf node represent one variable
        :param id: variable names, eg X1
        :param values: The values of the variable, eg X1=0, X1= 1 binary
        :param weights: The weights/probabilities of the values P(X1=0)= 0.2, P(X1=1) = 0.8

        '''
        assert isinstance(variable, (int,float,str)), 'One leaf node contains only one variable'
        assert len(domain) == len(weights),"domain and weights don't match."

        self.id = id
        self.domain = domain
        self.weights = weights
        self.scope = {variable}
        self.parents = parents

    # def eval(self, X):
    #     weights = self.weights/np.sum(self.weights)
    #     var = list(self.scope)[0]
    #     values = {k: w for k,w in zip(self.domain,weights)}
    #     return values[X[var]]

    def eval(self, X):
        var = list(self.scope)[0]
        if var in X:
            weights = self.weights/np.sum(self.weights)
            values = {k: w for k,w in zip(self.domain,weights)}
            return values[X[var]]
        else:
            return 1



if __name__ == '__main__':

    variables = ['x1','x2']

    s1 = LeafNode('s1','x1',weights=[2,8])
    s2 = LeafNode('s2', 'x1',weights=[1,9])
    s3 = LeafNode('s3', 'x2',weights=[4,6])

    p1 = ProdNode('p1 = s1+s3', [s1, s3])
    p2 = ProdNode('p2 = s2+s3', [s2, s3])

    s0 = SumNode('root', [p1,p2],[0.3,0.7])


    # 实现了简单的inference 过程
    # P(x1=T, x2 = F ) = 0.348
    X={'x1':0, 'x2':0}
    print(s0.eval(X))

    X={'x1':0, 'x2':1}
    print(s0.eval(X))

    X={'x1':1, 'x2':0}
    print(s0.eval(X))

    X={'x1':1, 'x2':1}
    print(s0.eval(X))

    # Queries 过程
    # P(x2=f| x1 = T ) = 0.4
    q={'x2':0,'x1':1}
    Q={'x1':1}
    print(s0.eval(q)/s0.eval(Q))

