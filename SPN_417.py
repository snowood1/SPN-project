from node_417 import *



class SPN():
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.leaves = {}
        self.root = None

    #Todo:
    def train(self):

        data = np.array([[0,'p'],
                         [1,'p'],
                         [0,'n'],
                         [1,'n']])

        names = ['x1','x2']

        for col, name in zip(data.transpose(), names):
            print(name,'\'s scope is',set(col))


            s1 = LeafNode('s1','x1',domain=[0,1] ,weights=[2,8])
            s2 = LeafNode('s2', 'x1',domain=[0,1] ,weights=[1,9])
            s3 = LeafNode('s3', 'x2',domain=['p', 'n'] ,weights=[4,6])

            p1 = ProdNode('p1 = s1+s3', [s1, s3])
            p2 = ProdNode('p2 = s2+s3', [s2, s3])

            s0 = SumNode('root', [p1,p2],[0.3,0.7])

        self.nodes=[s1,s2,s3,p1,p2,s0]
        self.root= s0

    def inference(self,variable,evidence=None):
        '''
        # P(X1=1,X2=0)= ?  Can output marginal and joint probability
        :param variable: input dict of the variables {x1:1, x2: 0}
        :return: probability (float)

        '''
        if evidence:
            return self.root.eval(variable)/self.root.eval(evidence)
        else:
            return self.root.eval(variable)

    #
    # def query(self,variable,evidence):
    #     '''
    #     return P (X|e)
    #     '''
    #




# 实现了简单的inference 过程
s=SPN()
s.train()

# P(x1=1, x2 = p ) = 0.348

X={'x1':1, 'x2':'p'}

print(s.inference(X))

# Queries 过程
# P(x2=f| x1 = T ) = 0.4
q={'x2':'p','x1':1}
Q={'x1':1}

print(s.inference(q,Q))
