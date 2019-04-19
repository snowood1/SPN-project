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

    def update_links(self):
        for child in self.children:
            self.scope = self.scope.union(child.scope)
            if self not in child.parents:
                child.add_parent(self)

    def eval(self):
        self.value = self.w @ [i.eval() for i in self.ch]
        return self.value


class ProductNode:
    def __init__(self, ch):
        self.ch = ch
        self.p = list()
        self.scope = set()
        self.value = None

    def eval(self):
        self.value = 1
        for i in self.ch:
            self.value *= i.eval()
        return self.value


class LeafNode:
    def __init__(self):
        self.value = None

    def eval(self):
        return self.value


class RVNode(SumNode):
    def __init__(self, rv, w=None):
        self.rv = rv

        ch = list()
        for x in rv.domain:
            ch.append(LeafNode())

        SumNode.__init__(self, ch, w)


class RV:
    def __init__(self, domain=(0,1)):
        self.domain = domain
        self.value = None
        self.instance = list()

    def set_value(self, value):
        self.value = value
        for i in self.instance:
            for j, d in zip(i.ch, self.domain):



if __name__ == '__main__':

    variables_name = ['x1','x2']

    Var=[]
    for k in range(2):
        Var.append(Variables(k))


    s1 = LeafNode('s1',Var[0],weights=[2,8])
    s2 = LeafNode('s2',Var[0] ,weights=[1,9])
    s3 = LeafNode('s3',Var[1],weights=[4,6])

    p1 = ProdNode('p1 = s1+s3', [s1, s3])
    p2 = ProdNode('p2 = s2+s3', [s2, s3])

    s0 = SumNode('root', [p1,p2],[0.3,0.7])


    # 实现了简单的inference 过程

    X=np.array([[0,0],     #P(x0=T, x1 = F )
                [0,1],
                [1,0],
                [1,1]])    #P(x0=F, x1 = T )

    print(s0.eval(X,[1,1]))  # [1,1] means P(x0, x1)

    print(s0.eval(X,[1,0]))  # [1,0] means P(x0) P(x0=1), P(x0=0)

    print(s0.eval(X,[0,1]))  # [0,1] means P(x1), P(x1=0) P(x1=1)

    # Queries 过程
    # P(x1| x0 ) = P(x0,x1)/P(x0)
    print(s0.eval(X,[1,1])/s0.eval(X,[1,0]))



s0.derivative = 1
s0.pass_gradient()
for s in s0.children:
    print(s.derivative)
