import numpy as np

class Node(object):
    def __init__(self, id, children=[],parents=[]):
        self.id = id
        self.parents = parents
        self.children = children
        self.scope= set()
        self.update_links()
        self.value=0
        self.derivative = 0

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

    def pass_gradient(self):
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

        #     for child
        # Todo:  MPE Inference (MPE assignment)  MPE 没有写，只有sum 模式

    def eval(self, X, Q, mpe=None):  #  X = {x1:0, x2:0 , x3:1}
        weights = self.weights/np.sum(self.weights)

        if mpe == None:
            s= np.dot(weights, [child.eval(X,Q) for child in self.children])
        self.value = s
        return self.value

    def pass_gradient(self):    # sum node 对child  pass gradient
        weights = self.weights/np.sum(self.weights)
        self.weights = weights + weights * self.derivate
        for child, w in self.children, self.weights:
            child.derivative = w

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

    def eval(self, X, Q):  #  X = {x1:0, x2:0 , x3:1}
        p = 1.0
        for child in self.children:
            p *= child.eval(X,Q)
        self.value = p
        return self.value

    def pass_gradient(self):    
        for child in self.children:
            child.derivative = child.derivative + self.derivative * \
                    np.prod([sib.value for sib in self.children if sib != child])

class LeafNode(Node):
    def __init__(self, id, variable,weights=[], parents=[]):

        '''
        One leaf node represent one variable
        :param id: variable names, eg X1
        :param values: The values of the variable, eg X1=0, X1= 1 binary
        :param weights: The weights/probabilities of the values P(X1=0)= 0.2, P(X1=1) = 0.8

        '''
        assert isinstance(variable, Variables), 'Input a variable class'
        assert len(weights)==len(variable.domain) or weights==[], 'Wrong dimension of weights'

        self.id = id
        self.variable =variable
        self.domain = variable.domain
        self.weights = weights if weights else np.ones(len(self.domain))
        self.scope = {variable.id}
        self.parents = parents
        self.value= 0
        self.derivate = 0

    def eval(self, X, Q):
        '''

        :param X:  means N X K data (np.array) with features x0 x1 x2 ...xk-1
        :param Q:  a vector represents which features are queried.
                    For example [0 1 0] means we only want to know P(x1)
                    [1,1,0] means P(x0,x1)
        :return:  np.array of values
        '''

        scope = list(self.scope)[0] #  The scope means this leaf nodes contain x?

        if Q[scope]==1:
            weights = self.weights/np.sum(self.weights)
            values = {k: w for k,w in zip(self.domain,weights)}
            self.value = np.vectorize(values.get)(X[:,scope])
            # print('self.value',self.value)
        else:
            self.value = np.ones(len(X))
            # print(' X self.value',self.value)

        return self.value

    def get_gradient(self):
        return
    def pass_gradient(self):
        weights = self.weights/np.sum(self.weights)
        self.weights = weights + weights * self.derivate


class Variables():
    def __init__(self,id,name='x',domain=[0,1]):
        self.id=id
        self.name = name+str(id)
        self.domain = domain
    def set_domain(self,domain):
        self.domain= domain



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



# s0.derivative = 1
# s0.pass_gradient()
# for s in s0.children:
#     print(s.derivative)
