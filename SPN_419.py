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
            Var.append(Variables(k))   # 定义了两个binary variables  x0, x1


        s1 = LeafNode('s1',Var[0],weights=[2,8])  # 叶子 s1 包含 x0
        s2 = LeafNode('s2',Var[0] ,weights=[1,9])  # s2 包含 x0
        s3 = LeafNode('s3',Var[1],weights=[4,6]) # s3 包含 x1
        s4 = LeafNode('s4',Var[1],weights=[3,7]) # s4 包含 x1

        p1 = ProdNode('p1 = s1+s3', [s1, s3])  # 第二层是product node  
        p2 = ProdNode('p2 = s2+s4', [s2, s4])

        s0 = SumNode('root', [p1,p2],[0.3,0.7]) # 第三层是root sum node

        self.nodes=[s1,s2,s3,s4, p1,p2,s0]
        self.root= s0

    def inference(self,X,query,evidence=None): 
        '''
        # P(Q|E)= ?  Can output marginal and joint probability

        :param X: training data
        :param query:  [1,1,0] means P(x0,x1)   
        :param evidence:  [1,0,0] means P(x0)
        :return:  marginal probability  or conditional probability
        '''
        if evidence:
            query = np.array(query) + np.array(evidence)        # P(x2| x1 ) = P(x2,x1 )/P(x1 )
            return self.root.eval(X,query)/self.root.eval(X,evidence)
        else:
            return self.root.eval(X,query)

    def gradient_descent(self):
        self.root.gradient = 1

        print('\nBefore gradient descent:')
        for node in self.nodes:
            print(node.id, node.gradient)

        self.root.pass_gradient()
        print('\nAfter gradient descent:')
        for node in self.nodes:
            print(node.id, node.gradient)



# 实现了简单的inference 过程
s=SPN()
s.train()

# X是training data  一共四行，每行两个feature x0 x1
#            x0 x1
X=np.array([[0,0],     #P(x0=0, x1 = 0 )
            [0,1],     #P(x0=0, x1 = 1 )
            [1,0],     #P(x0=1, x1 = 0 )
            [1,1]])    #P(x0=1, x1 = 1 )


print(s.inference(X,[1,1]))  # [1,1] means P(x0, x1) 意思是考虑x0,x1,联合概率


print(s.inference(X,[1,0]))  # [1,0] means P(x0) 意思是考虑x0, 不考虑 x1, marginal prob

print(s.inference(X,[0,1]))  # [0,1] means P(x1) 意思是考虑x1, 不考虑 x0, marginal prob

# P(x1| x0 )  第一个[0,1]表示查询P(x1), 第二个[1,0] 表示条件是P(x0)
print(s.inference(X,[0,1],[1,0]))

# P(x0| x1 )  第一个[1,0] 表示查询P(x0), 第二个[0,1] 表示条件是P(x1)
print(s.inference(X,[1,0],[0,1]))

# 现在我们看一下gradient decent的结果， 做gradient必须先inference了之后才能做gradient
s.inference(X,[1,1])
s.gradient_descent()
