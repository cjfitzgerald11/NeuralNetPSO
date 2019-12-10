import math
class Node:
    def __init__(self, typeNode):
        self.nodeTypes = {"input": 0, "bias": 1, "output": 2}
        self.nodeType = self.nodeTypes[typeNode]

    def activate(self, input):
        if self.nodeType == 0:
            return input
        elif self.nodeType == 1:
            return 1
        else:
            return self.sigmoid(input)

    def sigmoid(self,sum):
        return 1.0/(1.0 + math.exp(-sum + 0.5))

    def g_prime(self,sum):
        #g_dot = math.exp(0.5 - activation) / (((math.exp(0.5 - activation)) + 1) ** 2)
        g_dot = (1 - self.sigmoid(sum)) * self.sigmoid(sum)
        return g_dot
