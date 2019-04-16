import numpy as np

class Layer:
    def __init__(self, id, nodes=[]):
        self.id = id
        self.nodes = nodes
        self.check_nodes()

    def add_node(self, node):
        if type(node) == type(self.nodes[0]):
            if node not in self.nodes:
                self.nodes.append(node)
            else:
                print('Already exists')
        else:
            raise NameError('Adding a wrong class node')

    def check_nodes(self):
        nodes = self.nodes
        if not isinstance(nodes,list):
            self.nodes=[nodes]
        if all(isinstance(x,SumNode) for x in nodes) or \
                all(isinstance(x,ProdNode) for x in nodes) or \
                    all(isinstance(x,LeafNode) for x in nodes) or nodes==[]:
            return
        else:
            raise NameError('Layers should contain nodes of the same type')


class Node(object):
    def __init__(self, id, children=[],parents=[]):
        self.id = id
        self.parents = parents
        self.children = children
        self.value = 0.0

    def update_links(self):
        for child in self.children:
            if self.check_child(child):
                if self not in child.parents:
                    child.add_parent(self)
            else:
                raise NameError('Wrong child type')

    def add_child(self):
        pass

    def add_parent(self,parent):
        self.parents.append(parent)

    def check_child(self,node):   # 检查添加的child 是否正确
        pass

    def get_value(self):
        pass

class SumNode(Node):
    def __init__(self, id, children=[],weights=[],parents=[]):
        Node.__init__(self, id, children, parents)
        self.weights=weights   #  只有 sumnode 有 weights  对应
        self.update_links()

        self.normalize_weights()

    def add_child(self,node,weight=0):
        if self.check_child(node):
            self.children.append(node)
            node.add_parent(self)

        self.weights.append(weight)
        self.normalize_weights()

    def check_child(self, node):
        return isinstance(node,(ProdNode,LeafNode))  #  SumNode 只能接 ProdNode 和 LeafNode

    def set_weights(self,weights):
        self.weights=weights
        self.normalize_weights()

    def normalize_weights(self):
        if self.weights:
            self.weights = self.weights/np.sum(self.weights)
        else:
            l= len(self.children)
            self.weights = np.ones(l)/l    #  weights 默认 [1/n, 1/n, ... 1/n]  或者 随机？

    def get_value(self):
        self.value = 0.0
        for child, weight in zip(self.children, self.weights):
            self.value += weight * child.get_value()
        return self.value

class ProdNode(Node):
    def __init__(self, id, children=[],parents=[]):
        Node.__init__(self, id,children,parents)
        self.update_links()

    def add_child(self,node,weight=0):
        if self.check_child(node):
            self.children.append(node)
            node.add_parent(self)

    def check_child(self,node):
        return isinstance(node,SumNode)  #   Check 添加的child 是否是SumNode,

    def get_value(self):
        self.value = 1.0
        for child in self.children:
            self.value *= child.get_value()
        return self.value

class LeafNode(Node):
    def __init__(self, id,value=1.0,parents=[]):
        self.id = id
        self.value = value
        self.parents = parents

    def get_value(self):
        return self.value

if __name__ == '__main__':
    x1 = LeafNode('x1')
    x1_ = LeafNode('x1_',2)
    x2 = LeafNode('x2',3)
    x3 = LeafNode('x3',4)

    s1 = SumNode('s1=x1+x1_', [x1, x1_])
    s2 = SumNode('s2 = x2', [x2])

    p1 = ProdNode('p1 = s1+s2', [s1, s2])

    s0 = SumNode('s0=p1+x3', [p1,x3])

    s1.get_value()

