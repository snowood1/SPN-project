from node_419 import *
import pandas as pd

class SPN():
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.leaves = {}
        self.root = None


    #Todo:
    def train(self):

        Var=[]

        for k in range(2):
            Var.append(Variables(k))

        s1 = LeafNode('s1',Var[0] ,weights=[2,8])
        s2 = LeafNode('s2', Var[0],weights=[1,9])
        s3 = LeafNode('s3', Var[1],weights=[4,6])

        p1 = ProdNode('p1 = s1+s3', [s1, s3])
        p2 = ProdNode('p2 = s2+s3', [s2, s3])

        s0 = SumNode('root',[p1,p2],[0.3,0.7])

        self.nodes=[s1,s2,s3,p1,p2,s0]
        self.root= s0

    def inference(self,X,query,evidence=None):
        '''
        # P(Q|E)= ?  Can output marginal and joint probability

        :param X:
        :param query:  [1,1,0] means P(x0,x1)
        :param evidence:  [1,0,0] means P(x0)
        :return:  marginal probability  or conditional probability
        '''
        if evidence:
            query = np.array(query) + np.array(evidence)        # P(x2| x1 ) = P(x2,x1 )/P(x1 )
            return self.root.eval(X,query)/self.root.eval(X,evidence)
        else:
            return self.root.eval(X,query)



# 实现了简单的inference 过程
s=SPN()
s.train()
           # x0 x1
X=np.array([[0, 0],     #(x0=F, x1 = F )
            [0, 1],     #(x0=F, x1 = T )
            [1, 0],      #( x0 = T, x1= F
            [1, 1]])


print(s.inference(X,[1,1]))  # [1,1] means P(x0, x1) 考虑x0, x1

print(s.inference(X,[1,0]))  # [1,0] means P(x0) 不考虑x1

print(s.inference(X,[0,1]))  # [0,1] means P(x1) 不考虑x0

# P(x1| x0 )
print(s.inference(X,[0,1],[1,0]))    # Q= 只考虑x1,  evidence 只考虑x0

# P(x0| x1 )
print(s.inference(X,[1,0],[0,1]))   # Q 只考虑x0, evidence只考虑x1
